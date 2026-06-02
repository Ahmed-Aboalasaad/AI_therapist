from pydantic import BaseModel, Field
from typing import List, Dict, Any

class TranslationTestRequest(BaseModel):
    text: str = Field(..., description="Text to detect language and translate to English.")

class TranslationTestResponse(BaseModel):
    source_language: str = Field(..., description="Detected language.")
    translated_text: str = Field(..., description="Translated text in English.")

class TranslateBackTestRequest(BaseModel):
    text: str = Field(..., description="English text to translate back.")
    target_language: str = Field(..., description="The language to translate the text into.")

class TranslateBackTestResponse(BaseModel):
    translated_text: str = Field(..., description="Translated text.")

class IntentTestRequest(BaseModel):
    text: str = Field(..., description="Text to classify intent.")

class IntentTestResponse(BaseModel):
    intent: str = Field(..., description="Classified intent.")
    confidence: float = Field(..., description="Classification confidence.")

class EmotionTestRequest(BaseModel):
    text: str = Field(..., description="Text to classify emotion.")

class EmotionTestResponse(BaseModel):
    emotion: str = Field(..., description="Classified emotion.")
    confidence: float = Field(..., description="Classification confidence.")

class RagTestRequest(BaseModel):
    text: str = Field(..., description="Query to search context for in vector store.")

class RagTestResponse(BaseModel):
    query: str = Field(..., description="Original search query.")
    retrieved_contexts: List[Dict[str, Any]] = Field(..., description="List of retrieved contexts and source answers.")

class LanguageDetectionTestRequest(BaseModel):
    text: str = Field(..., description="Text to detect the language of.")

class LanguageDetectionTestResponse(BaseModel):
    detected_language: str = Field(..., description="The detected language.")
