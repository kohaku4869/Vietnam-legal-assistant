# file: vectorstore.py

import os
import faiss
import numpy as np
from langchain_core.documents import Document


class VectorDB:
    def __init__(self, embedding, persist_dir: str = "vectorstore"):
        """Quản lý vector stores và truy xuất tài liệu sử dụng FAISS.

        Args:
            embedding: Đối tượng có phương thức .embed_documents() và .embed_query().
            persist_dir (str): Thư mục lưu trữ vector stores (mặc định: 'vectorstore').
        """
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.embedding = embedding
        self.vectorstores = {}
        self.texts = {}

    def get_path(self, category: str):
        if not category or not isinstance(category, str) or any(c in category for c in '/\\'):
            raise ValueError("Category không hợp lệ.")
        return os.path.join(self.persist_dir, category)

    def get_index_path(self, category: str):
        return os.path.join(self.get_path(category), "index.faiss")

    def get_text_path(self, category: str):
        return os.path.join(self.get_path(category), "texts.txt")

    def load(self, category: str) -> bool:
        idx_path = self.get_index_path(category)
        txt_path = self.get_text_path(category)

        if not os.path.exists(idx_path) or not os.path.exists(txt_path):
            # Không cần in ra ở đây vì logic trong chat.py sẽ xử lý
            return False

        try:
            self.vectorstores[category] = faiss.read_index(idx_path)
            with open(txt_path, encoding="utf-8") as f:
                self.texts[category] = f.read().splitlines()
            return True
        except Exception as e:
            print(f"Lỗi khi tải vector store cho {category}: {e}")
            return False

    def build(self, category: str, documents):
        if not documents or not isinstance(documents, list) or not all(
                hasattr(doc, 'page_content') for doc in documents):
            raise ValueError("Documents không hợp lệ hoặc rỗng.")

        path = self.get_path(category)
        os.makedirs(path, exist_ok=True)

        try:
            texts = [doc.page_content for doc in documents]
            # SỬA Ở ĐÂY: Chuyển đổi kết quả thành numpy array một cách an toàn
            vectors_list = self.embedding.embed_documents(texts)
            vectors = np.asarray(vectors_list, dtype=np.float32)  # <--- THAY ĐỔI QUAN TRỌNG

            if vectors.ndim != 2:
                raise ValueError(f"Expected 2D array of vectors, but got shape {vectors.shape}")

            index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(vectors)

            self.vectorstores[category] = index
            self.texts[category] = texts
            faiss.write_index(index, self.get_index_path(category))
            with open(self.get_text_path(category), "w", encoding="utf-8") as f:
                for t in texts:
                    f.write(t.replace("\n", " ") + "\n")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi xây dựng vector store cho {category}: {e}")

    def query(self, query_vector: np.ndarray, category: str, top_k: int = 10) -> list[str]:
        if category not in self.vectorstores:
            if not self.load(category):
                raise ValueError(f"Vectorstore cho '{category}' không tồn tại hoặc không tải được.")

        # Đảm bảo query_vector là 2D array
        query_vector_2d = np.asarray([query_vector], dtype=np.float32)
        D, I = self.vectorstores[category].search(query_vector_2d, top_k)

        # I[0] chứa các chỉ số của các văn bản gần nhất
        return [self.texts[category][i] for i in I[0] if i != -1]

    def delete(self, category: str):
        if category in self.vectorstores:
            del self.vectorstores[category]
            del self.texts[category]
        path = self.get_path(category)
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)

    def update(self, category: str, new_documents):
        if category not in self.vectorstores:
            self.build(category, new_documents)
        else:
            current_docs = [Document(page_content=text) for text in self.texts[category]]
            all_docs = current_docs + new_documents
            # Xây dựng lại từ đầu với toàn bộ tài liệu
            self.build(category, all_docs)