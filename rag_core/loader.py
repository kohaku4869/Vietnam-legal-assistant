import os
import glob
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def remove_non_utf8(text: str) -> str:
    return ''.join(c for c in text if ord(c) < 128 or c == '\n')


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
            loader = PyPDFLoader(pdf_path)
            for doc in loader.load():
                doc.page_content = remove_non_utf8(doc.page_content)
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
