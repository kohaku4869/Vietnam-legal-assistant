from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

#Embedder
class Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        return np.array(self.embedding.embed_documents(texts))

    def embed_query(self, query):
        return np.array(self.embedding.embed_query(f"query: {query}"))