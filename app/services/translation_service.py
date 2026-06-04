import json
from pydantic import BaseModel, Field
import groq
from app.core.config import get_settings

settings = get_settings()

class TranslationResponse(BaseModel):
    source_language: str = Field(description="The detected language of the input text, e.g. 'Arabic', 'English', 'French', 'Spanish'.")
    translated_text: str = Field(description="The text translated into English. If the text was already in English, it should be kept identical to the input.")

class TranslationService:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.GROQ_MODEL_NAME
        self.client = groq.Groq(api_key=settings.GROQ_API_KEY)

    def detect_and_translate(self, text: str) -> TranslationResponse:
        """
        Detects the language of the text and translates it to English if it is not English.
        Uses Groq native JSON mode for reliability.
        """
        if not text.strip():
            return TranslationResponse(source_language="English", translated_text="")

        response = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert translator and language detector.\n"
                        "Analyze the user's input text.\n"
                        "1. Detect the language of the input text (e.g. 'Arabic', 'French', 'English').\n"
                        "2. Translate it to English if it is not English.\n"
                        "Respond ONLY in JSON format matching this schema:\n"
                        "{\n"
                        "  \"source_language\": \"detected language name\",\n"
                        "  \"translated_text\": \"the translated text in English, or the original text if it was already English\"\n"
                        "}"
                    )
                },
                {"role": "user", "content": text}
            ]
        )
        
        try:
            data = json.loads(response.choices[0].message.content)
            return TranslationResponse(
                source_language=data.get("source_language", "English"),
                translated_text=data.get("translated_text", text)
            )
        except Exception as e:
            # Fallback in case of JSON parse error
            return TranslationResponse(source_language="Unknown", translated_text=text)

    def translate_to_english(self, text: str, source_language: str) -> str:
        """
        Translates text from source_language to English.
        """
        if not text.strip() or source_language.lower() == "english" or source_language.lower() == "en":
            return text

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a professional translator.\n"
                        f"Translate the user's {source_language} text into English.\n"
                        f"Do NOT include any introduction, explanations, or quotes. Output ONLY the translated text."
                    )
                },
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()

    def translate_back(self, text: str, target_language: str) -> str:
        """
        Translates the English response back to the target language (e.g. Arabic).
        Uses plain text completion for maximum reliability.
        """
        if not text.strip() or target_language.lower() == "english":
            return text

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a professional translator.\n"
                        f"Translate the user's English text into {target_language}.\n"
                        f"Do NOT include any introduction, explanations, or quotes. Output ONLY the translated text."
                    )
                },
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
