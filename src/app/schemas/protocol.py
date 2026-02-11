from pydantic import BaseModel, Field
from typing import List

class ChatMessage(BaseModel):
    role: str
    content: str

class GenerationRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = Field(default=128, le=1024)
    temperature: float = Field(default=0.7, le=2.0)

class GenerationResponse(BaseModel):
    generated_text: str
    model: str