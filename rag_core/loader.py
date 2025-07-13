import os
import glob
import re
from typing import List
# THAY ĐỔI: Import thư viện mới
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

class Loader:
    def __init__(self, base_path: str = "data"):
        self.base_path = base_path

    def load_category(self, category: str) -> List[Document]:
        category_path = os.path.join(self.base_path, category)
        if not os.path.isdir(category_path):
            raise ValueError(f"Category folder not found: {category_path}")

        pdf_files = glob.glob(f"{category_path}/*.pdf")
        documents = []
        for pdf_path in pdf_files:
            # THAY ĐỔI: Sử dụng PyMuPDFLoader
            loader = PyMuPDFLoader(pdf_path)
            for doc in loader.load():
                # Chúng ta vẫn giữ lại bước dọn dẹp khoảng trắng
                doc.page_content = normalize_whitespace(doc.page_content)
                doc.metadata["category"] = category
                documents.append(doc)
        return documents

    def load_all(self) -> List[Document]:
        all_docs = []
        for category in os.listdir(self.base_path):
            full_path = os.path.join(self.base_path, category)
            if os.path.isdir(full_path):
                all_docs.extend(self.load_category(category))
        return all_docs