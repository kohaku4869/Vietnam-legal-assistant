import re
from operator import itemgetter
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class KeywordExtractionChain:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            """**Nhiệm vụ:** Bạn là một chuyên gia phân tích và mở rộng truy vấn pháp lý. Nhiệm vụ của bạn là nhận một câu hỏi bằng ngôn ngữ tự nhiên từ người dùng và chuyển đổi nó thành một danh sách các từ khóa và thuật ngữ pháp lý chính xác, có khả năng xuất hiện cao nhất trong các văn bản luật của Việt Nam.

            **Quy trình suy luận:**
            1.  Phân tích ý định thực sự của người dùng.
            2.  Không chỉ trích xuất từ trong câu hỏi. Hãy suy luận và mở rộng các thuật ngữ thông thường thành các thuật ngữ pháp lý tương đương.
            3.  Bổ sung các thuật ngữ liên quan trực tiếp (ví dụ: số Điều luật nếu có thể đoán được).

            **Các ví dụ:**
            -   **Câu hỏi gốc:** làm thế nào có thể từ đất nông nghiệp lên đất thổ cư?
                **Từ khóa đầu ra:** chuyển mục đích sử dụng đất, đất nông nghiệp, đất ở, đất thổ cư, quy hoạch sử dụng đất, thủ tục, chi phí
            -   **Câu hỏi gốc:** ly hôn thì con theo ai?
                **Từ khóa đầu ra:** ly hôn, quyền nuôi con, trông nom, chăm sóc, nuôi dưỡng, giáo dục con, Điều 81, thỏa thuận của cha mẹ
            -   **Câu hỏi gốc:** hàng xóm xây nhà lấn đất của tôi thì làm gì?
                **Từ khóa đầu ra:** tranh chấp đất đai, lấn đất, ranh giới thửa đất, hòa giải tranh chấp, giải quyết tranh chấp
            -   **Câu hỏi gốc:** sổ đỏ của tôi bị sai ngày sinh
                **Từ khóa đầu ra:** đính chính giấy chứng nhận, sai sót thông tin, cấp đổi giấy chứng nhận, thủ tục đính chính

            **QUY TẮC ĐẦU RA:**
            -   Chỉ trả về danh sách các từ khóa, phân cách bởi dấu phẩy.
            -   Không thêm bất kỳ lời giải thích nào khác.

            ---
            **Thực hiện với câu hỏi sau:**

            **Câu hỏi gốc:** "{question}"
            **Từ khóa đầu ra:**"""
        )
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser

    def invoke(self, question: str) -> list[str]:
        response = self.chain.invoke({"question": question})
        keywords = [k.strip() for k in response.split(',') if k.strip()]
        return keywords

class MultiQueryChain:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            """Bạn là một trợ lý AI hữu ích. Nhiệm vụ của bạn là nhận một câu hỏi của người dùng và tạo ra 3 phiên bản câu hỏi khác nhau có cùng ý nghĩa, nhưng sử dụng các cách diễn đạt hoặc từ khóa khác nhau để tìm kiếm trong một cơ sở dữ liệu pháp lý.
            QUY TẮC: Chỉ trả về các câu hỏi, mỗi câu trên một dòng, không có đánh số hay gạch đầu dòng.
            Câu hỏi gốc: {question}
            Các phiên bản câu hỏi khác:"""
        )
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser

    def invoke(self, question: str) -> list[str]:
        response = self.chain.invoke({"question": question})
        return [q.strip() for q in response.split('\n') if q.strip()]

class OfflineRAG:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            """**VAI TRÒ:** Bạn là một Luật sư AI, chuyên tư vấn về Luật dựa trên các văn bản pháp luật được cung cấp.

            **NGỮ CẢNH ĐẦU VÀO:**
            1.  **Lịch sử trò chuyện:** {chat_history}
            2.  **Ngữ cảnh được cung cấp:** Một danh sách các Điều luật trích từ Luật.
                <Ngữ cảnh>
                {context}
                </Ngữ cảnh>
            3.  **Câu hỏi của người dùng:** {question}

            **NHIỆM VỤ:**
            Viết một bài tư vấn chi tiết, rõ ràng, theo từng bước để trả lời câu hỏi của người dùng. **TUYỆT ĐỐI KHÔNG** chỉ nói "hãy tham khảo điều X". Bạn phải **giải thích nội dung** của điều đó và nó áp dụng vào trường hợp của người dùng như thế nào.

            **QUY TRÌNH SUY LUẬN BẮT BUỘC (Hãy nghĩ trong đầu trước khi viết câu trả lời):**
            1.  **Phân tích câu hỏi:** Xác định các ý chính người dùng muốn biết. Ví dụ: với câu hỏi "chuyển đất nông nghiệp sang đất ở", các ý chính là: (a) có được phép không?, (b) điều kiện là gì?, (c) thủ tục các bước ra sao?, (d) cơ quan nào giải quyết?
            2.  **Sàng lọc Ngữ cảnh:** Đọc qua các Điều luật trong Ngữ cảnh. Gán mỗi Điều luật cho một hoặc nhiều ý chính đã xác định ở bước 1. Ví dụ: Điều 121 trả lời cho ý (a), Điều 116 trả lời cho ý (b), Điều 227 trả lời cho ý (c), Điều 123 trả lời cho ý (d).
            3.  **Lập dàn ý cho câu trả lời:** Dựa trên các điều luật đã sàng lọc, cấu trúc lại câu trả lời một cách logic. Ví dụ:
                -   Mở đầu: Khẳng định việc này là có thể nhưng phải được phép.
                -   Thân bài 1 - Điều kiện: Trình bày các điều kiện dựa trên Điều 116.
                -   Thân bài 2 - Thủ tục: Trình bày các bước dựa trên Điều 227.
                -   Thân bài 3 - Thẩm quyền: Nêu rõ cơ quan giải quyết dựa trên Điều 123.
                -   Kết luận: Nhắc nhở về nghĩa vụ tài chính và các lưu ý khác.

            **SAU KHI ĐÃ SUY NGHĨ THEO CÁC BƯỚC TRÊN, HÃY VIẾT CÂU TRẢ LỜI HOÀN CHỈNH VÀO MỤC DƯỚI ĐÂY.**

            **Câu trả lời tư vấn:**
            """
        )
        self.parser = StrOutputParser()
        self.chain = self._build_chain()

    def _build_chain(self):
        input_data = {
            "context": itemgetter("context") | RunnableLambda(self.format_docs),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history") | RunnableLambda(self.format_history)
        }
        return input_data | self.prompt | self.llm | self.parser

    def invoke(self, docs, question: str, chat_history: list):
        return self.chain.invoke({"context": docs, "question": question, "chat_history": chat_history})

    def stream(self, docs, question: str, chat_history: list):
        # 1. Chuẩn bị dữ liệu đầu vào cho prompt
        inputs = {
            "context": self.format_docs(docs),
            "question": question,
            "chat_history": self.format_history(chat_history)
        }

        full_prompt = self.prompt.format_prompt(**inputs).to_string()

        return self.llm.stream(full_prompt)
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_history(self, chat_history: list) -> str:
        if not chat_history: return "Không có lịch sử trò chuyện."
        history = []
        for msg in chat_history:
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý AI"
            history.append(f"{role}: {msg['content']}")
        return "\n".join(history)