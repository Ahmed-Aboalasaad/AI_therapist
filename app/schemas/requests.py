from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    message: str = Field(..., description="The message/query sent by the user.")
    session_id: Optional[str] = Field(default="default-session", description="The conversational session identifier for history retrieval.")
