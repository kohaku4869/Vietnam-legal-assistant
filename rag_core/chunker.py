import re
from typing import List
from langchain_core.documents import Document


class Chunker:
    """
    Chia nhỏ văn bản luật theo từng "Điều".
    Đây là phương pháp hiệu quả cho các văn bản có cấu trúc.
    """

    def split(self, documents: List[Document]) -> List[Document]:
        full_text = "\n".join([doc.page_content for doc in documents])

        # Sử dụng biểu thức chính quy để tìm các "Điều" làm điểm bắt đầu của mỗi chunk.
        # Mẫu này sẽ tìm "Điều" theo sau là một số và một dấu chấm.
        chunks_text = re.split(r'(Điều \d+\..*?)\n', full_text, flags=re.DOTALL)

        # Xử lý kết quả từ re.split để ghép tiêu đề và nội dung
        final_chunks = []
        for i in range(1, len(chunks_text), 2):
            # Mỗi chunk sẽ bao gồm tiêu đề (vd: "Điều 81.") và nội dung tiếp theo của nó
            chunk_content = (chunks_text[i] + chunks_text[i + 1]).strip()
            if len(chunk_content) > 50:  # Lọc bỏ các chunk quá ngắn hoặc rỗng
                final_chunks.append(Document(page_content=chunk_content))

        if not final_chunks:
            # Nếu không chia được theo Điều, dùng cách cũ để dự phòng
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            return text_splitter.split_documents(documents)

        return final_chunks