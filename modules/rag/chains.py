from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from modules.rag.helpers.config import get_settings
from modules.rag.database import get_retriever
from modules.rag.prompts import MENTAL_HEALTH_SYSTEM_PROMPT, GENERAL_SYSTEM_PROMPT

settings = get_settings()

def format_docs(docs):
    formatted_context = ""
    MAX_CHAR_LIMIT = 4000

    for i, doc in enumerate(docs, 1):
        real_answer = doc.metadata.get('answers', 'No response available')

        current_reference = (
            f"--- Reference {i} ---\n"
            f"Patient Context: {doc.page_content}\n"
            f"Expert Answer from Dataset:\n{real_answer}\n\n"
        )

        if len(formatted_context) + len(current_reference) > MAX_CHAR_LIMIT:
            formatted_context += "--- [Remaining references omitted for length and token safety] ---\n"
            break

        formatted_context += current_reference

    return formatted_context

def retrieve_context(query: str):
    """
    Retrieve matching documents from Qdrant and format them into a context string.
    Returns: (formatted_context_string, list_of_langchain_documents)
    """
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return format_docs(docs), docs

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


if __name__ == "__main__":
    try:
        print("Constructing LLM Chain and testing a live query on Groq...")
        chain = get_mental_health_chain()
        
        test_query = "I am feeling extremely anxious and hopeless because of my growing debts. What should I do?"
        print(f"\n🗣️ User: {test_query}")
        print("🤖 Thinking...")
        
        response = chain.invoke({
            "prompt": test_query,
            "emotion": "anxiety",
            "language": "English",
            "context": "Expert advice: seek debt consolidation and counseling.",
            "chat_history": ""
        })
        print(f"\n🌟 AI Counselor:\n{response}\n")
        print("Chain Layer works flawlessly!")
        
    except Exception as e:
        print(f"Chain Layer Test Failed: {str(e)}")