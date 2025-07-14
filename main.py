import streamlit as st
import os
from dotenv import load_dotenv
import time

# Import các thành phần cốt lõi từ dự án của bạn
from rag_core.loader import Loader
from rag_core.chunker import Chunker
from rag_core.embedder import Embedder
from rag_core.vectorstore import VectorDB
from rag_core.reranker import Reranker
from rag_core.llm import GoogleLLMPipeline
from rag_core.rag_chain import OfflineRAG, MultiQueryChain, KeywordExtractionChain
from langchain_core.documents import Document

# --- CÀI ĐẶT TRANG VÀ CSS TÙY CHỈNH ---

st.set_page_config(
    page_title="Trợ lý Pháp lý AI",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="auto"
)

# CSS tùy chỉnh để giao diện đẹp hơn
st.markdown("""
<style>
    /* Chỉnh sửa bong bóng chat của assistant (bot) */
    [data-testid="stChatMessage"][data-testid="chatAvatarIcon-assistant"] + div [data-testid="stMarkdown"] {
        background-color: #262730; /* Màu nền tối hơn một chút */
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #3a3b4d;
    }

    /* Chỉnh sửa bong bóng chat của user */
    [data-testid="stChatMessage"][data-testid="chatAvatarIcon-user"] + div [data-testid="stMarkdown"] {
        background-color: #404258; /* Màu nền cho tin nhắn của người dùng */
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #474e68;
    }

    /* Chỉnh sửa thanh sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E1F29;
    }
</style>
""", unsafe_allow_html=True)

# --- ÁNH XẠ CATEGORY VÀ KHỞI TẠO PIPELINE ---

CATEGORY_MAP = {
    "family_law": "Luật Hôn nhân và Gia đình",
    "land_law": "Luật Đất đai",
    "business_law": "Luật Doanh nghiệp",
    "civil_law": "Bộ luật Dân sự",
    "criminal_law": "Bộ luật Hình sự"
}


@st.cache_data
def load_api_keys():
    load_dotenv()
    return os.getenv("GOOGLE_API_KEY")


GOOGLE_API_KEY = load_api_keys()


@st.cache_resource
def initialize_rag_pipeline():
    print("--- Đang khởi tạo RAG Pipeline ---")
    if not GOOGLE_API_KEY:
        st.error("Không tìm thấy GOOGLE_API_KEY.")
        st.stop()

    loader = Loader(base_path="data")
    chunker = Chunker()
    embedder = Embedder(model_name="intfloat/multilingual-e5-large")
    vector_db = VectorDB(embedder)
    # Quay lại Reranker local để đảm bảo miễn phí và ổn định khi deploy
    reranker = Reranker(model_name="BAAI/bge-reranker-large")
    llm = GoogleLLMPipeline(api_key=GOOGLE_API_KEY, model_name="gemini-1.5-pro")
    rag_chain = OfflineRAG(llm=llm)
    multi_query_chain = MultiQueryChain(llm=llm)
    keyword_chain = KeywordExtractionChain(llm=llm)
    return loader, chunker, vector_db, reranker, rag_chain, multi_query_chain, keyword_chain, embedder


# Khởi tạo các thành phần
loader, chunker, vector_db, reranker, rag_chain, multi_query_chain, keyword_chain, embedder = initialize_rag_pipeline()

# --- GIAO DIỆN NGƯỜI DÙNG ---

with st.sidebar:
    st.title("⚖️ Trợ lý Pháp lý")
    st.markdown("Chào mừng bạn đến với chatbot tư vấn pháp lý. Vui lòng chọn một lĩnh vực và đặt câu hỏi.")

    display_names = list(CATEGORY_MAP.values())
    selected_display_name = st.selectbox(
        "Lĩnh vực pháp lý:",
        options=display_names,
    )
    selected_category = [key for key, value in CATEGORY_MAP.items() if value == selected_display_name][0]

    st.markdown("---")
    if st.button("Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- LOGIC XỬ LÝ CHAT ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các câu hỏi gợi ý nếu cuộc trò chuyện trống
if not st.session_state.messages:
    st.info("Bắt đầu cuộc trò chuyện bằng cách đặt một câu hỏi hoặc chọn một trong các gợi ý dưới đây.")
    example_questions = {
        "land_law": "Làm thế nào để chuyển đổi đất nông nghiệp sang đất thổ cư?",
        "family_law": "Ly hôn thì con dưới 3 tuổi sẽ theo ai?",
        "business_law": "Thủ tục thành lập công ty TNHH một thành viên?",
        "civil_law": "Thời hiệu khởi kiện về thừa kế tài sản là bao lâu?",
        "criminal_law": "Tội trộm cắp tài sản bị xử lý như thế nào?"
    }

    cols = st.columns(2)
    questions_to_show = [q for cat, q in example_questions.items() if cat == selected_category]
    if not questions_to_show:
        questions_to_show = list(example_questions.values())[:4]

    for i, question in enumerate(questions_to_show[:4]):
        if cols[i % 2].button(question, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Nguồn tham khảo"):
                for source in message["sources"]:
                    st.info(source)

# Xử lý input mới từ người dùng
if prompt := st.chat_input("Hỏi điều bạn muốn biết..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Suy nghĩ..."):
            try:
                # Build VectorDB nếu cần
                if not vector_db.load(selected_category):
                    status_placeholder = st.empty()
                    status_placeholder.info(
                        f"Lần đầu truy cập, đang xây dựng cơ sở dữ liệu cho '{selected_display_name}'...")
                    docs = loader.load_category(selected_category)
                    chunks = chunker.split(docs)
                    vector_db.build(selected_category, chunks)
                    status_placeholder.success(f"Xây dựng cơ sở dữ liệu thành công!")
                    time.sleep(2)
                    status_placeholder.empty()

                # Quy trình RAG nâng cao
                queries_generated = multi_query_chain.invoke(prompt)
                all_queries = [prompt] + queries_generated
                keywords_extracted = keyword_chain.invoke(prompt)

                semantic_candidates = []
                for q in all_queries:
                    q_vec = embedder.embed_query(q)
                    candidates = vector_db.query(q_vec, selected_category, top_k=5)
                    semantic_candidates.extend(candidates)
                semantic_candidates = list(dict.fromkeys(semantic_candidates))[:20]

                all_texts = vector_db.texts.get(selected_category, [])
                keyword_candidates = []
                if keywords_extracted:
                    for text in all_texts:
                        if any(keyword.lower() in text.lower() for keyword in keywords_extracted):
                            keyword_candidates.append(text)
                keyword_candidates = keyword_candidates[:20]

                combined_candidates = list(dict.fromkeys(semantic_candidates + keyword_candidates))
                reranked_chunks = reranker.rerank(prompt, combined_candidates, top_n=5)  # Lấy top 5 cho LLM

                reranked_docs = [Document(page_content=txt) for txt in reranked_chunks]
                chat_history_for_prompt = list(st.session_state.messages[:-1])
                answer = rag_chain.invoke(reranked_docs, prompt, chat_history=chat_history_for_prompt)

                st.markdown(answer)
                # Hiển thị nguồn tham khảo bên dưới câu trả lời
                if reranked_chunks:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": reranked_chunks})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {e}")
                answer = f"Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn: {e}"
                st.session_state.messages.append({"role": "assistant", "content": answer})