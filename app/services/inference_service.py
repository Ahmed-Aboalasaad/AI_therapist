from langchain_groq import ChatGroq
from app.core.config import get_settings

settings = get_settings()

class InferenceService:
    """
    Service responsible for LangChain chain execution, LLM invocation, and response generation.
    """
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL_NAME,
            temperature=0.4,
            max_tokens=1500
        )

    def generate_response(self, prompt: str) -> str:
        """
        Invokes the LLM with the fully constructed prompt text and returns the response.
        """
        response = self.llm.invoke(prompt)
        return response.content.strip()
