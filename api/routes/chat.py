from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
from typing import List, Tuple

from rag_core.loader import Loader
from rag_core.chunker import Chunker
from rag_core.embedder import Embedder
from rag_core.retriever import Retriever
from rag_core.reranker import Reranker
from rag_core.llm import GoogleLLMPipeline
from rag_core.rag_chain import OfflineRAG
from langchain_core.documents import Document

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    category: str
    history: List[Tuple[str, str]] = []

class QueryResponse(BaseModel):
    chunks: List[str]
    answer: str

# Load environment variables
# API_KEY = os.getenv("GOOGLE_API_KEY")
# if not API_KEY:
#     raise RuntimeError("GOOGLE_API_KEY not set in environment")
API_KEY="AIzaSyAEU3qeU7UpDQTg3gbbAajSijFqRefkRPM"

# Initialize core components
loader = Loader()
chunker = Chunker()
embedder = Embedder(model_name="intfloat/multilingual-e5-large")
reranker = Reranker()
llm = GoogleLLMPipeline(api_key=API_KEY, model_name="gemini-1.5-flash")
rag = OfflineRAG(llm=llm)

# Cache for retrievers by category
retrievers = {}

@router.post("/query", response_model=QueryResponse)
async def chat_query(req: QueryRequest):
    category = req.category

    if category not in retrievers:
        retriever = Retriever(dim=1024)
        if not retriever.load(category):
            docs = loader.load_category(category)
            chunks = chunker.split(docs)
            texts = [c.page_content for c in chunks]
            vectors = embedder.embed_documents(texts)
            retriever.build(vectors, texts, category)
        retrievers[category] = retriever

    retriever = retrievers[category]
    q_vec = embedder.embed_query(req.question)
    candidates = retriever.query(q_vec, top_k=10)
    reranked = reranker.rerank(req.question, candidates)[:5]

    # Fake Document objects for LangChain
    docs = [Document(page_content=txt) for txt in reranked]

    # Optional: include history in prompt if needed
    answer = rag.invoke({
        "question": req.question,
        "context": "\n\n".join(d.page_content for d in docs)
    })

    return QueryResponse(
        chunks=[doc.page_content for doc in docs],
        answer=answer
    )
