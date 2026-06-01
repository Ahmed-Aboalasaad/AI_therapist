import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from modules.rag.chains import get_mental_health_chain

router = APIRouter(
    prefix="/api",
    tags=["Chat"]
)

try:
    print("[Router] Initializing RAG Chain components for Chat Route...")
    rag_chain = get_mental_health_chain()
    print("[Router] Chat AI Chain loaded successfully!")
except Exception as e:
    print(f"[Router] Failed to initialize RAG Chain: {str(e)}")
    raise e

class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
   
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    try:
        async def response_generator():
            try:
                for chunk in rag_chain.stream(request.message):
                    yield chunk
                    await asyncio.sleep(0.01)
            except Exception as stream_err:
                print(f"Error during streaming chunks: {str(stream_err)}")
                yield "\n[An error occurred during response generation]"

        return StreamingResponse(response_generator(), media_type="text/plain")
        
    except Exception as e:
        print(f"Error during chat invocation: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="An error occurred while processing your request through the AI layers."
        )
    
