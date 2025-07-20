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


st.set_page_config(
    page_title="Trợ lý Pháp lý AI",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown("""
<style>
    /* Chỉnh sửa bong bóng chat của assistant (bot) */
    [data-testid="stChatMessage"][data-testid="chatAvatarIcon-assistant"] + div [data-testid="stMarkdown"] {
        background-color: #262730; /* Màu nền tối hơn một chút */
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #3a3b4d;
    }

    /* Chỉnh sửa bong bóng chat của người dùng */
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
    # Ưu tiên đọc từ st.secrets khi deploy, nếu không thì dùng .env
    return st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))


GOOGLE_API_KEY = load_api_keys()


@st.cache_resource
def initialize_rag_pipeline():
    """
    Hàm này chỉ chạy một lần để khởi tạo tất cả các thành phần nặng.
    """
    print("--- Đang khởi tạo RAG Pipeline ---")
    if not GOOGLE_API_KEY:
        st.error("Không tìm thấy GOOGLE_API_KEY. Vui lòng thiết lập biến này.")
        st.stop()

    loader = Loader(base_path="data")
    chunker = Chunker()
    embedder = Embedder()
    vector_db = VectorDB(embedder)
    reranker = Reranker()
    llm = GoogleLLMPipeline(api_key=GOOGLE_API_KEY, model_name="gemini-2.5-flash")
    rag_chain = OfflineRAG(llm=llm)
    multi_query_chain = MultiQueryChain(llm=llm)
    keyword_chain = KeywordExtractionChain(llm=llm)
    return loader, chunker, vector_db, reranker, rag_chain, multi_query_chain, keyword_chain, embedder


loader, chunker, vector_db, reranker, rag_chain, multi_query_chain, keyword_chain, embedder = initialize_rag_pipeline()


with st.sidebar:
    st.title("⚖️ Trợ lý Pháp lý")
    st.markdown("Chào mừng! Vui lòng chọn một lĩnh vực và đặt câu hỏi bên dưới.")

    display_names = list(CATEGORY_MAP.values())

    is_chat_started = len(st.session_state.get('messages', [])) > 0

    default_index = 0
    if 'selected_display_name' in st.session_state:
        try:
            default_index = display_names.index(st.session_state.selected_display_name)
        except ValueError:
            default_index = 0

    selected_display_name = st.selectbox(
        "Lĩnh vực pháp lý:",
        options=display_names,
        index=default_index,
        disabled=is_chat_started,
        key='selected_display_name'
    )

    selected_category = [key for key, value in CATEGORY_MAP.items() if value == selected_display_name][0]

    st.markdown("---")
    if st.button("Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.clear()
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Bắt đầu bằng cách đặt một câu hỏi hoặc chọn một trong các gợi ý dưới đây.")
    example_questions = {
        "land_law": "Làm thế nào để chuyển đổi đất nông nghiệp sang đất thổ cư?",
        "family_law": "Thủ tục ly hôn và quyền nuôi con được quy định ra sao?",
        "business_law": "Thủ tục thành lập công ty TNHH một thành viên?",
        "civil_law": "Thời hiệu khởi kiện về thừa kế tài sản là bao lâu?",
        "criminal_law": "Tội trộm cắp tài sản bị xử lý như thế nào?"
    }

    question_to_show = example_questions.get(selected_category, "Tôi có thể giúp gì cho bạn?")
    if st.button(question_to_show, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": question_to_show})
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Xem nguồn tham khảo"):
                for i, source in enumerate(message["sources"]):
                    st.info(f"Nguồn {i + 1}:\n\n{source}")

if prompt := st.chat_input("Hỏi điều bạn muốn biết..."):
    if not st.session_state.messages:
        st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_prompt = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        with st.spinner("Đang thực hiện quy trình tìm kiếm và tổng hợp nâng cao..."):
            try:
                if not vector_db.load(selected_category):
                    st.error(
                        f"Lỗi: Không tìm thấy file index cho '{selected_display_name}'. Vui lòng chạy `build_indexes.py` trước.")
                    st.stop()

                queries = [last_prompt] + multi_query_chain.invoke(last_prompt)
                keywords = keyword_chain.invoke(last_prompt)

                semantic_candidates = []
                for q in queries:
                    q_vec = embedder.embed_query(q)
                    semantic_candidates.extend(vector_db.query(q_vec, selected_category, top_k=7))

                all_texts = vector_db.texts.get(selected_category, [])
                keyword_candidates = []
                if keywords:
                    for text in all_texts:
                        if any(kw.lower() in text.lower() for kw in keywords):
                            keyword_candidates.append(text)

                combined_candidates = list(dict.fromkeys(semantic_candidates + keyword_candidates))[:40]
                reranked_chunks = reranker.rerank(last_prompt, combined_candidates)[:10]

                reranked_docs = [Document(page_content=txt) for txt in reranked_chunks]

                chat_history_for_prompt = list(st.session_state.messages[:-1])
                answer = rag_chain.invoke(reranked_docs, last_prompt, chat_history=chat_history_for_prompt)

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": reranked_chunks})

            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {e}")
                answer = "Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn."
                st.session_state.messages.append({"role": "assistant", "content": answer})