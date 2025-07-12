
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import List, Tuple

from rag_core.loader import Loader
from rag_core.chunker import Chunker
from rag_core.embedder import Embedder
from rag_core.vectorstore import VectorDB
from rag_core.reranker import Reranker
from rag_core.llm import GoogleLLMPipeline
from rag_core.rag_chain import OfflineRAG
from langchain_core.documents import Document

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set in environment")
router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    category: str
    history: List[Tuple[str, str]] = []

class QueryResponse(BaseModel):
    chunks: List[str]
    answer: str

# Initialize core components
loader = Loader()
chunker = Chunker()
embedder = Embedder(model_name="intfloat/multilingual-e5-large")
vector_db = VectorDB(embedder)
reranker = Reranker()
llm = GoogleLLMPipeline(api_key=API_KEY, model_name="gemini-1.5-flash")
rag = OfflineRAG(llm=llm)

@router.post("/query", response_model=QueryResponse)
async def chat_query(req: QueryRequest):
    print("\n--- BẮT ĐẦU YÊU CẦU MỚI ---")
    # Kiểm tra đầu vào
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if not req.category or any(c in req.category for c in '/\\'):
        raise HTTPException(status_code=400, detail="Invalid category name")

    category = req.category
    print(f"DEBUG: Category nhận được: {category}")

    # --- LUỒNG LOGIC QUAN TRỌNG ---
    print("DEBUG: Đang kiểm tra xem có cần build vector store không...")
    load_result = vector_db.load(category)
    print(f"DEBUG: Kết quả của vector_db.load('{category}') là: {load_result}")

    if not load_result:
        print("DEBUG: Vector store chưa tồn tại hoặc không tải được. Bắt đầu quá trình build.")
        try:
            # Bước 1: Tải tài liệu
            print("DEBUG: Đang gọi loader.load_category...")
            docs = loader.load_category(category)
            print(f"DEBUG: Tải thành công {len(docs)} tài liệu.")

            # Bước 2: Chia nhỏ tài liệu
            print("DEBUG: Đang gọi chunker.split...")
            chunks = chunker.split(docs)
            print(f"DEBUG: Chia thành công thành {len(chunks)} chunks.")

            # Bước 3: Build vector store
            print("DEBUG: Đang gọi vector_db.build...")
            vector_db.build(category, chunks)
            print("DEBUG: Build vector store THÀNH CÔNG.")

        except Exception as e:
            print(f"!!! LỖI NGHIÊM TRỌNG trong quá trình build: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to build vector store for {category}: {str(e)}")
    else:
        print("DEBUG: Đã tải vector store từ file có sẵn. Bỏ qua bước build.")

    # --- TIẾP TỤC XỬ LÝ ---
    print("DEBUG: Đang tạo vector cho câu hỏi...")
    q_vec = embedder.embed_query(req.question)
    print("DEBUG: Đang truy vấn các ứng viên từ VectorDB...")
    candidates = vector_db.query(q_vec, category, top_k=10)
    print(f"DEBUG: Tìm thấy {len(candidates)} ứng viên.")

    print("DEBUG: Đang rerank các ứng viên...")
    reranked = reranker.rerank(req.question, candidates)[:5]
    docs = [Document(page_content=txt) for txt in reranked]
    print(f"DEBUG: Đã rerank còn {len(docs)} tài liệu tốt nhất.")

    print("DEBUG: Đang gọi RAG chain để sinh câu trả lời...")
    answer = rag.invoke(docs, req.question)
    print("DEBUG: Đã nhận được câu trả lời từ LLM.")
    print("--- KẾT THÚC YÊU CẦU ---")

    return QueryResponse(
        chunks=[doc.page_content for doc in docs],
        answer=answer
    )