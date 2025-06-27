import os
import faiss
import numpy as np

class Retriever:
    def __init__(self, dim: int, persist_dir: str = "vectorstore"):
        self.dim = dim
        self.persist_dir = persist_dir
        self.index = None
        self.texts = []

    def get_index_path(self, category: str):
        return os.path.join(self.persist_dir, category, "index.faiss")

    def get_text_path(self, category: str):
        return os.path.join(self.persist_dir, category, "texts.txt")

    def load(self, category: str) -> bool:
        idx_path = self.get_index_path(category)
        txt_path = self.get_text_path(category)

        if not os.path.exists(idx_path) or not os.path.exists(txt_path):
            return False

        self.index = faiss.read_index(idx_path)
        with open(txt_path, encoding="utf-8") as f:
            self.texts = f.read().splitlines()
        return True

    def build(self, vectors: np.ndarray, texts: list[str], category: str):
        index = faiss.IndexFlatL2(self.dim)
        index.add(np.asarray(vectors, dtype=np.float32))
        self.index = index
        self.texts = texts

        # Lưu index và texts
        os.makedirs(os.path.join(self.persist_dir, category), exist_ok=True)
        faiss.write_index(self.index, self.get_index_path(category))
        with open(self.get_text_path(category), "w", encoding="utf-8") as f:
            for t in texts:
                f.write(t.replace("\n", " ") + "\n")

    def query(self, vector: np.ndarray, top_k: int = 5) -> list[str]:
        D, I = self.index.search(np.asarray([vector], dtype=np.float32), top_k)
        return [self.texts[i] for i in I[0]]