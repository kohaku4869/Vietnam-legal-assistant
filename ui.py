import streamlit as st
import os
from dotenv import load_dotenv

# Import các thành phần cốt lõi từ dự án của bạn
from rag_core.loader import Loader
from rag_core.chunker import Chunker
from rag_core.embedder import Embedder
from rag_core.vectorstore import VectorDB
from rag_core.reranker import Reranker
from rag_core.llm import GoogleLLMPipeline
from rag_core.rag_chain import OfflineRAG
from langchain_core.documents import Document

# --- CÀI ĐẶT CHUNG & KHỞI TẠO ---

st.set_page_config(
    page_title="Trợ lý Pháp lý AI",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="auto"
)

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("Vui lòng thiết lập GOOGLE_API_KEY trong file .env của bạn.")
    st.stop()


@st.cache_resource
def initialize_rag_pipeline():
    print("--- Đang khởi tạo RAG Pipeline (chỉ chạy một lần) ---")
    loader = Loader(base_path="data")
    chunker = Chunker()
    embedder = Embedder(model_name="intfloat/multilingual-e5-large")
    vector_db = VectorDB(embedder)
    reranker = Reranker()
    llm = GoogleLLMPipeline(api_key=API_KEY, model_name="gemini-2.5-flash")
    rag = OfflineRAG(llm=llm)
    return loader, chunker, vector_db, reranker, rag, embedder


loader, chunker, vector_db, reranker, rag, embedder = initialize_rag_pipeline()

# --- GIAO DIỆN NGƯỜI DÙNG ---

st.title("⚖️ Trợ lý Pháp lý AI")
st.caption("Hỏi đáp các vấn đề pháp lý dựa trên dữ liệu bạn cung cấp.")

with st.sidebar:
    st.header("Tùy chọn")
    try:
        available_categories = [d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))]
        if not available_categories:
            st.warning("Không tìm thấy thư mục lĩnh vực nào trong 'data'.")
            st.stop()
    except FileNotFoundError:
        st.error("Thư mục 'data' không tồn tại. Vui lòng tạo và thêm dữ liệu.")
        st.stop()

    selected_category = st.selectbox(
        "Chọn lĩnh vực pháp lý:",
        options=available_categories,
        index=0
    )
    st.markdown("---")
    # Thêm nút để xóa lịch sử chat
    if st.button("Bắt đầu cuộc trò chuyện mới"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chào bạn, tôi có thể giúp gì về vấn đề pháp lý của bạn?"}]
        st.rerun()  # Tải lại trang để xóa hiển thị

    st.markdown("---")
    st.info("Ứng dụng này đang trong giai đoạn phát triển.", icon="ℹ️")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Chào bạn, tôi có thể giúp gì về vấn đề pháp lý của bạn?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Đặt câu hỏi của bạn ở đây..."):
    # Lấy lịch sử chat TRƯỚC khi thêm tin nhắn mới
    # Chúng ta chỉ muốn gửi các tin nhắn cũ làm ngữ cảnh
    chat_history_for_prompt = list(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Vui lòng chờ, tôi đang tìm kiếm và suy nghĩ..."):
            try:
                if not vector_db.load(selected_category):
                    st.info(f"Vector store cho '{selected_category}' chưa có, bắt đầu xây dựng...")
                    docs = loader.load_category(selected_category)
                    chunks = chunker.split(docs)
                    vector_db.build(selected_category, chunks)
                    st.success(f"Xây dựng vector store cho '{selected_category}' thành công!")

                q_vec = embedder.embed_query(prompt)
                candidates = vector_db.query(q_vec, selected_category, top_k=15)
                reranked_chunks = reranker.rerank(prompt, candidates)[:5]
                reranked_docs = [Document(page_content=txt) for txt in reranked_chunks]

                # THAY ĐỔI QUAN TRỌNG: Gửi lịch sử chat vào hàm invoke
                answer = rag.invoke(reranked_docs, prompt, chat_history=chat_history_for_prompt)

                st.markdown(answer)
                with st.expander("Xem các nguồn thông tin đã tham khảo"):
                    for i, chunk in enumerate(reranked_chunks):
                        st.info(f"Nguồn {i + 1}:\n\n{chunk}")

            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {e}")
                answer = f"Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})