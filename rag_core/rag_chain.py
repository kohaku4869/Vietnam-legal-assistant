from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import re


class StrOutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    def extract_answer(self, text_response: str, pattern: str = r"Answer :\s*(.*)") -> str:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text_response


class OfflineRAG:
    def __init__(self, llm):
        self.llm = llm
        # THAY ĐỔI 1: Cập nhật Prompt Template để có thêm "chat_history"
        self.prompt = PromptTemplate.from_template(
            """Bạn là một trợ lý pháp lý AI hữu ích. Trả lời câu hỏi của người dùng dựa trên ngữ cảnh và lịch sử trò chuyện được cung cấp.
            Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết. Đừng cố bịa ra câu trả lời.

            Lịch sử trò chuyện:
            {chat_history}

            Ngữ cảnh được cung cấp:
            {context}

            Câu hỏi của người dùng: {question}

            Câu trả lời của bạn:"""
        )
        self.parser = StrOutputParser()
        self.chain = self._build_chain()

    def _build_chain(self):
        # THAY ĐỔI 2: Thêm "chat_history" vào chuỗi xử lý
        input_data = {
            "context": itemgetter("context") | RunnableLambda(self.format_docs),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history") | RunnableLambda(self.format_history)
        }
        return input_data | self.prompt | self.llm | self.parser

    # THAY ĐỔI 3: Cập nhật hàm invoke để nhận thêm "chat_history"
    def invoke(self, docs, question: str, chat_history: list):
        return self.chain.invoke({
            "context": docs,
            "question": question,
            "chat_history": chat_history
        })

    def format_docs(self, docs):
        """Định dạng danh sách tài liệu thành một chuỗi duy nhất."""
        return "\n\n".join(doc.page_content for doc in docs)

    # THAY ĐỔI 4: Thêm hàm mới để định dạng lịch sử chat
    def format_history(self, chat_history: list) -> str:
        """Định dạng lịch sử chat thành một chuỗi cho prompt."""
        if not chat_history:
            return "Không có lịch sử trò chuyện."

        formatted_history = []
        for message in chat_history:
            role = "Người dùng" if message["role"] == "user" else "Trợ lý AI"
            formatted_history.append(f"{role}: {message['content']}")

        return "\n".join(formatted_history)