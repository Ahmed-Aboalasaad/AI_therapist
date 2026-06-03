import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first before importing any app modules
load_dotenv(Path(__file__).resolve().parent / ".env")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8")
    ]
)

from app.api.chat import router as chat_router
from app.api.test_endpoints import router as test_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(
    title="AI Therapist Chat Flow API",
    description="Full-flow mental health support chatbot API incorporating Groq translation, intent/emotion classification, and Qdrant RAG.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routers
app.include_router(chat_router)
app.include_router(test_router)

# Mount the static files directory directly at the root /
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "description": "AI Therapist Flow System is online."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
