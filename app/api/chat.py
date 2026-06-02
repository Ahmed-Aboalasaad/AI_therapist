from fastapi import APIRouter, HTTPException
from app.schemas.requests import ChatRequest
from app.schemas.responses import ChatResponse
from app.services.orchestrator import FlowOrchestrator

router = APIRouter(
    prefix="/api",
    tags=["Chat"]
)

# Instantiate the FlowOrchestrator once on startup to pre-load the models
try:
    orchestrator = FlowOrchestrator()
except Exception as e:
    # We will log the error but not crash import time, so that uvicorn can load and show startup logs
    print(f"Error loading FlowOrchestrator: {e}")
    orchestrator = None

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if orchestrator is None:
        raise HTTPException(
            status_code=500,
            detail="System is not fully initialized. Please check that GROQ_API_KEY is configured in the environment."
        )

    try:
        result = orchestrator.process_query(request.message)
        return ChatResponse(**result)
    except Exception as e:
        print(f"Error processing query in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))