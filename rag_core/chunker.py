import re
from typing import List
from langchain_core.documents import Document


class Chunker:
    def split(self, documents: List[Document]) -> List[Document]:
        # Gộp nội dung từ tất cả các trang thành một chuỗi văn bản duy nhất
        full_text = "\n".join([doc.page_content for doc in documents])

        # Regex mới: Tách tại vị trí ngay trước "Điều X."
        # Ký hiệu `(?=...)` là một "positive lookahead", nó tìm vị trí để tách
        # mà không "ăn" mất chính chuỗi "Điều X.", giúp nó luôn ở đầu mỗi chunk.
        chunks_text = re.split(r'(?=Điều \d+\.)', full_text)

        final_chunks = []
        for chunk_content in chunks_text:
            cleaned_chunk = chunk_content.strip()
            # Chỉ lấy các chunk có nội dung thực sự (dài hơn 50 ký tự) để loại bỏ
            # các phần rác hoặc phần giới thiệu ngắn ở đầu văn bản.
            if len(cleaned_chunk) > 50 and cleaned_chunk.startswith("Điều"):
                final_chunks.append(Document(page_content=cleaned_chunk))

        return final_chunks