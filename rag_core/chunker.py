from typing import List
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings

class Chunker:
    """
    Split documents into semantically coherent chunks.
    """
    def __init__(self,
                 # model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 model_name: str = "intfloat/multilingual-e5-large",
                 chunk_size: int = 500,
                 threshold_type: str = "percentile",
                 threshold_amount: int = 95):
        embeddings = SentenceTransformerEmbeddings(model_name=model_name)
        self.splitter = SemanticChunker(
            embeddings=embeddings,
            buffer_size=1,
            breakpoint_threshold_type = threshold_type,
            breakpoint_threshold_amount=threshold_amount,
            min_chunk_size=chunk_size,
            add_start_index=True,
        )

    def split(self, documents) -> List:
        return self.splitter.split_documents(documents)