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
vector_db = VectorDB(embedder.embedding)
reranker = Reranker()
llm = GoogleLLMPipeline(api_key=API_KEY, model_name="gemini-1.5-flash")
rag = OfflineRAG(llm=llm)

@router.post("/query", response_model=QueryResponse)
async def chat_query(req: QueryRequest):
    # Kiểm tra đầu vào
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if not req.category or any(c in req.category for c in '/\\'):
        raise HTTPException(status_code=400, detail="Invalid category name")

    category = req.category
    if not vector_db.load(category):
        try:
            docs = loader.load_category(category)
            chunks = chunker.split(docs)
            vector_db.build(category, chunks)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to build vector store for {category}: {str(e)}")

    q_vec = embedder.embed_query(req.question)
    candidates = vector_db.query(q_vec, category, top_k=10)
    reranked = reranker.rerank(req.question, candidates)[:5]
    docs = [Document(page_content=txt) for txt in reranked]

    answer = rag.invoke(docs, req.question)

    return QueryResponse(
        chunks=[doc.page_content for doc in docs],
        answer=answer
    )