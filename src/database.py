from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from src.helpers.config import get_settings
from src.helpers.data_helpers import load_embedding_model

settings = get_settings()

def get_vector_store() -> QdrantVectorStore:
    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY
    )

    embeddings = load_embedding_model()
    
    return QdrantVectorStore(
        client=client,
        collection_name=settings.COLLECTION_NAME,
        embedding=embeddings
    )

def get_retriever():
    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 2,              
            "score_threshold": 0.70  
        }
    )
    return retriever


if __name__ == "__main__":
    try:
        print("Testing DB Connection and Retriever initialization...")
        retriever = get_retriever()
        print("DB Layer is configured perfectly! Ready to be injected into the LLM Chain.")
    except Exception as e:
        print(f"DB Layer Test Failed: {str(e)}")