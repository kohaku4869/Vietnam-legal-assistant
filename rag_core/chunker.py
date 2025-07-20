import re
from typing import List
from langchain_core.documents import Document


class Chunker:
    def split(self, documents: List[Document]) -> List[Document]:
        full_text = "\n".join([doc.page_content for doc in documents])

        chunks_text = re.split(r'(?=Điều \d+\.)', full_text)

        final_chunks = []
        for chunk_content in chunks_text:
            cleaned_chunk = chunk_content.strip()
            if len(cleaned_chunk) > 50 and cleaned_chunk.startswith("Điều"):
                final_chunks.append(Document(page_content=cleaned_chunk))

        return final_chunks