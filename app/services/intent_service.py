from typing import Literal
import groq
import instructor
from pydantic import BaseModel, Field
from app.core.config import get_settings

settings = get_settings()

intent_category = Literal["greeting", "goodbye", "gratitude", "asking_mental_health_question", "out_of_scope"]

class IntentResponse(BaseModel):
    intent: intent_category = Field(description="The strictly classified intent of the user's query.")
    confidence: float = Field(description="Confidence score for this classification, between 0.0 and 1.0")

class IntentService:
    def __init__(self):
        self.client = instructor.from_groq(groq.Groq(api_key=settings.GROQ_API_KEY))

    def classify_user_intent(self, user_query: str) -> IntentResponse:
        """
        Classifies the user's query into predefined categories using Groq and Instructor.
        """
        response = self.client.chat.completions.create(
            model=settings.GROQ_MODEL_NAME,
            response_model=IntentResponse,
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are an expert intent classifier for a mental health chatbot. "
                        "Classify the user's input into one of the predefined categories.\n\n"
                        "Here are some examples to guide you:\n"
                        "- User: 'سلام عليكم، صبّاح الخير' -> intent: 'greeting', confidence: 1.0\n"
                        "- User: 'hello there, good morning' -> intent: 'greeting', confidence: 1.0\n"
                        "- User: 'أنا بحس بـ خنقة وضيق في التنفس بقالي يومين' -> intent: 'asking_mental_health_question', confidence: 0.97\n"
                        "- User: 'I feel very anxious and can't sleep' -> intent: 'asking_mental_health_question', confidence: 0.98\n"
                        "- User: 'تسلم يا غالي شكراً ليك' -> intent: 'gratitude', confidence: 1.0\n"
                        "- User: 'أقرب مطعم بيتزا فين؟' -> intent: 'out_of_scope', confidence: 1.0\n"
                    )
                },
                {"role": "user", "content": user_query}
            ]
        )
        return response