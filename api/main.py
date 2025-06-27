from fastapi import FastAPI
from api.routes.chat import router as chat_router

app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation chatbot with Google Gemini",
    version="0.1.0",
)

app.include_router(chat_router, prefix="/chat", tags=["chat"])