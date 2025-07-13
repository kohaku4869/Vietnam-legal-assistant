import streamlit as st
import os
from dotenv import load_dotenv

from rag_core.loader import Loader
from rag_core.chunker import Chunker
from rag_core.embedder import Embedder
from rag_core.vectorstore import VectorDB
from rag_core.reranker import Reranker
from rag_core.llm import GoogleLLMPipeline
from rag_core.rag_chain import OfflineRAG, MultiQueryChain, KeywordExtractionChain
from langchain_core.documents import Document

st.set_page_config(page_title="Trợ lý Pháp lý AI", page_icon="⚖️", layout="wide")

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")


@st.cache_resource
def initialize_rag_pipeline():
    print("--- Đang khởi tạo RAG Pipeline ---")
    loader = Loader(base_path="data")
    chunker = Chunker()
    embedder = Embedder(model_name="intfloat/multilingual-e5-large")
    vector_db = VectorDB(embedder)
    reranker = Reranker(model_name="BAAI/bge-reranker-large")
    llm = GoogleLLMPipeline(api_key=API_KEY, model_name="gemini-2.5-flash")
    rag_chain = OfflineRAG(llm=llm)
    multi_query_chain = MultiQueryChain(llm=llm)
    keyword_chain = KeywordExtractionChain(llm=llm)
    return loader, chunker, vector_db, reranker, rag_chain, multi_query_chain, keyword_chain, embedder


if not API_KEY:
    st.error("Vui lòng thiết lập GOOGLE_API_KEY trong file .env của bạn.")
    st.stop()

loader, chunker, vector_db, reranker, rag_chain, multi_query_chain, keyword_chain, embedder = initialize_rag_pipeline()

st.title("⚖️ Trợ lý Pháp lý AI")
st.caption("Hỏi đáp các vấn đề pháp lý dựa trên dữ liệu bạn cung cấp.")

with st.sidebar:
    st.header("Tùy chọn")
    try:
        available_categories = [d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))]
    except FileNotFoundError:
        st.error("Thư mục 'data' không tồn tại.")
        st.stop()

    selected_category = st.selectbox("Chọn lĩnh vực pháp lý:", options=available_categories, index=0)
    st.markdown("---")
    if st.button("Bắt đầu trò chuyện mới"):
        st.session_state.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Chào bạn, tôi có thể giúp gì về vấn đề pháp lý của bạn?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Đặt câu hỏi của bạn ở đây..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Vui lòng chờ, tôi đang thực hiện quy trình tìm kiếm nâng cao..."):
            try:
                if not vector_db.load(selected_category):
                    with st.status(f"Bắt đầu xây dựng Vector Store cho '{selected_category}'...",
                                   expanded=True) as status:
                        docs = loader.load_category(selected_category);
                        status.write("Đã tải tài liệu...")
                        chunks = chunker.split(docs);
                        status.write("Đã chia nhỏ văn bản...")
                        vector_db.build(selected_category, chunks);
                        status.write("Đã xây dựng index...")
                        status.update(label="Xây dựng hoàn tất!", state="complete")

                queries_generated = multi_query_chain.invoke(prompt)
                all_queries = [prompt] + queries_generated
                keywords_extracted = keyword_chain.invoke(prompt)

                semantic_candidates = []
                for q in all_queries:
                    q_vec = embedder.embed_query(q)
                    candidates = vector_db.query(q_vec, selected_category, top_k=5)
                    semantic_candidates.extend(candidates)

                semantic_candidates = list(dict.fromkeys(semantic_candidates))[:20]

                # 2.2. Tìm kiếm từ khóa
                all_texts = vector_db.texts.get(selected_category, [])
                keyword_candidates = []
                if keywords_extracted:
                    for text in all_texts:
                        if any(keyword.lower() in text.lower() for keyword in keywords_extracted):
                            keyword_candidates.append(text)

                keyword_candidates = keyword_candidates[:20]

                combined_candidates = list(dict.fromkeys(semantic_candidates + keyword_candidates))

                with st.expander(f"DEBUG: {len(combined_candidates)} ứng viên được đưa vào Reranker"):
                    st.json(combined_candidates)

                reranked_chunks = reranker.rerank(prompt, combined_candidates)[:10]

                with st.expander("DEBUG: Top 10 nguồn thông tin cuối cùng được sử dụng"):
                    st.json(reranked_chunks)

                reranked_docs = [Document(page_content=txt) for txt in reranked_chunks]
                chat_history_for_prompt = list(st.session_state.messages[:-1])
                answer = rag_chain.invoke(reranked_docs, prompt, chat_history=chat_history_for_prompt)
                st.markdown(answer)

            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {e}")
                answer = f"Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})