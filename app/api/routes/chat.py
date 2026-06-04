from fastapi import APIRouter, HTTPException
from app.schemas.requests import ChatRequest
from app.schemas.responses import ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(
    prefix="/api",
    tags=["Chat"]
)

# Instantiate ChatService once on startup
try:
    chat_service = ChatService()
except Exception as e:
    print(f"Error loading ChatService: {e}")
    chat_service = None

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if chat_service is None:
        raise HTTPException(
            status_code=500,
            detail="System is not fully initialized. Please check that GROQ_API_KEY is configured in the environment."
        )

    try:
        result = chat_service.process_query(request.message, session_id=request.session_id)
        return ChatResponse(**result)
    except Exception as e:
        print(f"Error processing query in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
