from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

class StrOutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    def extract_answer(self, text_response: str,
                       pattern: str = r"Answer :\s*(.*)") -> str:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text_response

class OfflineRAG:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            "Sử dụng thông tin sau để trả lời câu hỏi.\n\nContext:\n{context}\n\nQuestion: {question}"
        )
        self.parser = StrOutputParser()

    def invoke(self, retriever):
        input_data = {
            "context": RunnableLambda(lambda x: x["question"]) | retriever | self.format_docs,
            "question": RunnablePassthrough()
        }
        rag_chain = (
            input_data
            | self.prompt
            | self.llm
            | self.parser
        )
        return rag_chain

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)