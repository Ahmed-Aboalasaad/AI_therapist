from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from modules.rag.helpers.config import get_settings
from modules.rag.database import get_retriever
from modules.rag.prompts import MENTAL_HEALTH_SYSTEM_PROMPT

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

def get_mental_health_chain():
    retriever = get_retriever()

    llm = ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL_NAME, 
        temperature=0.4,
        max_tokens=500
    )

    prompt = ChatPromptTemplate.from_template(MENTAL_HEALTH_SYSTEM_PROMPT)

    rag_chain = (
        {
            "context": retriever | format_docs, 
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    try:
        print("Constructing RAG Chain and testing a live query on Groq...")
        chain = get_mental_health_chain()
        
        test_query = "I am feeling extremely anxious and hopeless because of my growing debts. What should I do?"
        print(f"\n🗣️ User: {test_query}")
        print("🤖 Thinking...")
        
        response = chain.invoke(test_query)
        print(f"\n🌟 AI Counselor:\n{response}\n")
        print("Chain Layer works flawlessly!")
        
    except Exception as e:
        print(f"Chain Layer Test Failed: {str(e)}")