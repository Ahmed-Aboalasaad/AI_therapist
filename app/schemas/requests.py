from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., description="The message/query sent by the user.")
