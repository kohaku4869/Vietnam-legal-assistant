import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDB:
    def __init__(self, persist_dir: str = "vectorstore"):
        self.persist_dir = persist_dir
        self.embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        self.vectorstores = {}  # Lưu theo category

    def get_path(self, category: str):
        return os.path.join(self.persist_dir, category)

    def load(self, category: str) -> bool:
        path = self.get_path(category)
        if not os.path.exists(path):
            return False
        self.vectorstores[category] = FAISS.load_local(path, self.embedding)
        return True

    def build(self, category: str, documents):
        path = self.get_path(category)
        os.makedirs(path, exist_ok=True)
        db = FAISS.from_documents(documents, self.embedding)
        db.save_local(path)
        self.vectorstores[category] = db

    def get_retriever(self, category: str, k: int = 5):
        if category not in self.vectorstores:
            raise ValueError(f"Vectorstore for category '{category}' is not loaded.")
        return self.vectorstores[category].as_retriever(search_kwargs={"k": k})
