from langchain_qdrant import QdrantVectorStore
from app.repositories.qdrant_repository import QdrantRepository
from app.rag.embeddings import get_vector_store

def get_retriever(qdrant_repo: QdrantRepository):
    """
    Constructs and returns the MMR retriever for search queries.
    """
    vector_store = get_vector_store(qdrant_repo)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,              
            "score_threshold": 0.70  
        }
    )
    return retriever

def format_docs(docs):
    """
    Formats the list of documents into a single expert context block.
    """
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

def retrieve_context(query: str, qdrant_repo: QdrantRepository):
    """
    Retrieve matching documents from Qdrant and format them into a context string.
    Returns: (formatted_context_string, list_of_langchain_documents)
    """
    retriever = get_retriever(qdrant_repo)
    docs = retriever.invoke(query)
    return format_docs(docs), docs
