'''Fast API endpoint. NO AI Logic.. Just call the orchestrator'''

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes import chat
from src.helpers.config import get_settings
settings = get_settings()
app = FastAPI(
    title="Financial Mental Health Chatbot API",
    description="Production-ready RAG API with Standard Root Architecture",
    version="1.0.0"
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "model": settings.GROQ_MODEL_NAME, "vector_store": "qdrant"}