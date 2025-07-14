# ⚖️ Chatbot Tư vấn Pháp lý sử dụng RAG

Đây là dự án xây dựng một chatbot thông minh, có khả năng trả lời các câu hỏi phức tạp về luật pháp Việt Nam dựa trên các văn bản luật được cung cấp. Chatbot sử dụng kiến trúc **Retrieval-Augmented Generation (RAG)** tiên tiến để đảm bảo câu trả lời vừa chính xác, vừa bám sát vào nguồn tài liệu gốc, giảm thiểu tối đa hiện tượng "ảo giác" (hallucination) của các mô hình ngôn ngữ lớn (LLM).

## ✨ Tính năng nổi bật

-   **Giao diện Chat trực quan:** Xây dựng bằng Streamlit, cho phép người dùng tương tác một cách dễ dàng và thân thiện.
-   **Hỗ trợ đa lĩnh vực:** Người dùng có thể chọn lĩnh vực pháp lý (ví dụ: Luật Đất đai, Luật Hôn nhân và Gia đình) để đặt câu hỏi.
-   **Ghi nhớ ngữ cảnh hội thoại:** Chatbot có khả năng hiểu các câu hỏi nối tiếp dựa trên lịch sử trò chuyện.
-   **Kiến trúc Retrieval Tiên tiến:**
    -   **Multi-Query Retriever:** Tự động tạo nhiều phiên bản câu hỏi từ câu hỏi gốc của người dùng để tăng độ bao phủ khi tìm kiếm.
    -   **Keyword Extraction:** Tự động dùng LLM để phân tích câu hỏi và trích xuất các từ khóa pháp lý quan trọng.
    -   **Hybrid Search:** Kết hợp cả tìm kiếm ngữ nghĩa (Semantic Search) và tìm kiếm từ khóa (Keyword Search) để cho ra danh sách tài liệu ứng viên toàn diện nhất.
    -   **Advanced Reranker:** Sử dụng model `BAAI/bge-reranker-large` mạnh mẽ để sàng lọc và chọn ra những ngữ cảnh liên quan nhất trước khi đưa cho LLM.
-   **Tối ưu hóa Dữ liệu đầu vào:**
    -   Sử dụng loader `PyMuPDF` để trích xuất văn bản từ PDF một cách sạch sẽ.
    -   Áp dụng phương pháp Chunking thông minh, chia nhỏ văn bản luật theo từng "Điều" để đảm bảo tính toàn vẹn của ngữ cảnh pháp lý.
-   **Chế độ Debug:** Tích hợp các mục hiển thị gỡ lỗi ngay trên giao diện, cho phép theo dõi quá trình làm việc nội bộ của hệ thống (các câu hỏi được tạo, từ khóa, các ứng viên được tìm thấy...).

## 🏗️ Kiến trúc và Công nghệ

-   **Ngôn ngữ chính:** Python
-   **Giao diện người dùng (UI):** Streamlit
-   **Mô hình ngôn ngữ lớn (LLM):** Google Gemini (`gemini-1.5-pro` hoặc `gemini-1.5-flash`)
-   **Framework RAG:** LangChain
-   **Model Embedding:** `intfloat/multilingual-e5-large`
-   **Model Reranker:** `BAAI/bge-reranker-large`
-   **Vector Store:** FAISS (chạy trên CPU)
-   **Xử lý PDF:** PyMuPDF

## ⚙️ Cài đặt

Để chạy dự án trên máy của bạn, hãy làm theo các bước sau:

**1. Tải mã nguồn:**
```bash
git clone <URL_KHO_CHỨA_CỦA_BẠN>
cd <TÊN_THƯ_MỤC_DỰ_ÁN>
```

**2. (Khuyến khích) Tạo và kích hoạt môi trường ảo:**
- Trên Windows:
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
- Trên macOS/Linux:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

**3. Cài đặt các thư viện cần thiết:**
Dự án đã có file `requirements.txt`. Chạy lệnh sau để cài đặt tất cả trong một lần:
```bash
pip install -r requirements.txt
```

**4. Cấu hình API Key:**
- Tạo một file mới ở thư mục gốc của dự án và đặt tên là `.env`.
- Mở file `.env` và thêm vào dòng sau, thay thế `YOUR_API_KEY` bằng khóa API của bạn từ Google AI Studio:
  ```
  GOOGLE_API_KEY="YOUR_API_KEY"
  ```

**5. Thêm dữ liệu:**
- Tạo một thư mục tên là `data` ở thư mục gốc.
- Bên trong `data`, tạo các thư mục con tương ứng với từng lĩnh vực pháp lý (ví dụ: `family_law`, `land_law`).
- Đặt các file PDF luật liên quan vào các thư mục con tương ứng.

## 🚀 Khởi chạy ứng dụng

Sau khi hoàn tất các bước cài đặt, chạy lệnh sau trong terminal:

```bash
streamlit run ui.py
```

Ứng dụng sẽ tự động mở trong trình duyệt của bạn.

## 📁 Cấu trúc thư mục

```
/
├── data/
│   ├── family_law/
│   │   └── luat_hon_nhan_gia_dinh.pdf
│   └── land_law/
│       └── luat_dat_dai.pdf
├── rag_core/
│   ├── chunker.py      # Logic chia nhỏ văn bản
│   ├── embedder.py     # Logic vector hóa
│   ├── llm.py          # Logic gọi LLM
│   ├── loader.py       # Logic tải và dọn dẹp file
│   ├── rag_chain.py    # Logic của các chain RAG
│   ├── reranker.py     # Logic sắp xếp lại
│   └── vectorstore.py  # Logic quản lý Vector Store (FAISS)
├── vectorstore/        # (Thư mục này sẽ được tự động tạo)
├── .env                # File chứa API key (cần tự tạo)
├── requirements.txt    # Danh sách các thư viện cần thiết
└── ui.py               # File để chạy giao diện Streamlit
```