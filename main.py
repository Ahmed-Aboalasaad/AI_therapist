import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve root path and load environment variables before importing any app modules
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.logging import setup_logging
from app.api.routes.chat import router as chat_router

# Setup centralized logging
setup_logging()

app = FastAPI(
    title="AI Therapist Chat Flow API",
    description="Restructured clean-architecture mental health support chatbot API incorporating translation, classification, and Qdrant RAG.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the restructured router
app.include_router(chat_router)

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "description": "AI Therapist Restructured Flow System is online."
    }

# Mount the static files directory at the root /
# Must be mounted AFTER all other API endpoints to avoid shadowing them
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
