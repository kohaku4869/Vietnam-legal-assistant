from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, query):
        return self.model.encode([f"query: {query}"], convert_to_numpy=True)[0]