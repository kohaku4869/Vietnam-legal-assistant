import os
import faiss
import numpy as np
from langchain_core.documents import Document

class VectorDB:
    def __init__(self, embedding, persist_dir: str = "vectorstore"):
        """Quản lý vector stores và truy xuất tài liệu sử dụng FAISS.

        Args:
            embedding: Hàm embedding để chuyển đổi văn bản thành vector.
            persist_dir (str): Thư mục lưu trữ vector stores (mặc định: 'vectorstore').
        """
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.embedding = embedding
        self.vectorstores = {}  # Lưu trữ FAISS index và texts cho từng category
        self.texts = {}  # Lưu trữ danh sách texts cho từng category

    def get_path(self, category: str):
        """Tạo đường dẫn cho category."""
        if not category or not isinstance(category, str) or any(c in category for c in '/\\'):
            raise ValueError("Category không hợp lệ.")
        return os.path.join(self.persist_dir, category)

    def get_index_path(self, category: str):
        """Tạo đường dẫn cho file index FAISS."""
        return os.path.join(self.get_path(category), "index.faiss")

    def get_text_path(self, category: str):
        """Tạo đường dẫn cho file texts."""
        return os.path.join(self.get_path(category), "texts.txt")

    def load(self, category: str) -> bool:
        """Tải vector store từ thư mục cho category.

        Args:
            category (str): Tên category.

        Returns:
            bool: True nếu tải thành công, False nếu không.
        """
        idx_path = self.get_index_path(category)
        txt_path = self.get_text_path(category)

        if not os.path.exists(idx_path) or not os.path.exists(txt_path):
            print(f"Vector store path or index file does not exist: {idx_path}")
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
        """Xây dựng vector store cho category từ danh sách documents.

        Args:
            category (str): Tên category.
            documents: Danh sách các Document từ langchain_core.documents.
        """
        if not documents or not isinstance(documents, list) or not all(hasattr(doc, 'page_content') for doc in documents):
            raise ValueError("Documents không hợp lệ hoặc rỗng.")

        path = self.get_path(category)
        os.makedirs(path, exist_ok=True)

        try:
            # Chuyển đổi documents thành vectors
            texts = [doc.page_content for doc in documents]
            vectors = self.embedding.embed_documents(texts)
            index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(np.asarray(vectors, dtype=np.float32))

            # Lưu index và texts
            self.vectorstores[category] = index
            self.texts[category] = texts
            faiss.write_index(index, self.get_index_path(category))
            with open(self.get_text_path(category), "w", encoding="utf-8") as f:
                for t in texts:
                    f.write(t.replace("\n", " ") + "\n")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi xây dựng vector store cho {category}: {e}")

    def query(self, query_vector: np.ndarray, category: str, top_k: int = 10) -> list[str]:
        """Tìm kiếm top_k tài liệu gần nhất với query_vector trong category.

        Args:
            query_vector (np.ndarray): Vector của câu hỏi.
            category (str): Tên category.
            top_k (int): Số lượng tài liệu trả về.

        Returns:
            list[str]: Danh sách nội dung các tài liệu.
        """
        if category not in self.vectorstores:
            if not self.load(category):
                raise ValueError(f"Vectorstore cho '{category}' không tồn tại hoặc không tải được.")
        D, I = self.vectorstores[category].search(np.asarray([query_vector], dtype=np.float32), top_k)
        return [self.texts[category][i] for i in I[0]]

    def delete(self, category: str):
        """Xóa vector store cho category."""
        if category in self.vectorstores:
            del self.vectorstores[category]
            del self.texts[category]
        path = self.get_path(category)
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)

    def update(self, category: str, new_documents):
        """Cập nhật vector store với các tài liệu mới."""
        if category not in self.vectorstores:
            self.build(category, new_documents)
        else:
            current_docs = [Document(page_content=text) for text in self.texts[category]]
            all_docs = current_docs + new_documents
            self.build(category, all_docs)