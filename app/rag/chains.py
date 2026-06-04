from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import get_settings
from app.rag.prompts import MENTAL_HEALTH_SYSTEM_PROMPT, GENERAL_SYSTEM_PROMPT

settings = get_settings()

def get_mental_health_chain(intent: str = "asking_mental_health_question"):
    """
    Returns a decoupled LLM chain with a prompt dynamically chosen based on the intent.
    For 'asking_mental_health_question', expects: prompt, emotion, language, context, chat_history.
    For other intents, expects: prompt, emotion, language, chat_history.
    """
    llm = ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL_NAME, 
        temperature=0.4,
        max_tokens=1500
    )

    template = MENTAL_HEALTH_SYSTEM_PROMPT if intent == "asking_mental_health_question" else GENERAL_SYSTEM_PROMPT
    prompt = ChatPromptTemplate.from_template(template)

    llm_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return llm_chain
