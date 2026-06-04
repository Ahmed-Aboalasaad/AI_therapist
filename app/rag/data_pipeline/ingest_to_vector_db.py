import os
import sys
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

# Append the project root to sys.path so it runs directly as a script
sys.path.append(str(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))))

from app.rag.data_pipeline.data_preprocessing import run_preprocessing
from app.core.config import get_settings
from app.rag.embeddings import load_embedding_model

settings = get_settings()

def convert_df_to_documents(df: pd.DataFrame) -> list[Document]:
    print(f"Converting {len(df)} DataFrame rows into LangChain Documents...")
    docs = []
    for idx, row in df.iterrows():
        doc = Document(
            page_content=str(row["Context"]), 
            metadata={
                "answers": str(row["Response"]),
                "source": "mental_health_counseling_dataset"
            }
        )
        docs.append(doc)
    return docs

def ingest_data():
    print("Base Pipeline Core initialized.")

    run_preprocessing()

    embeddings = load_embedding_model()

    print("\nConnecting to Qdrant Database...")
    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY
    )

    try:
        collection_info = client.get_collection(collection_name=settings.COLLECTION_NAME)
        if collection_info.points_count > 0:
            print(f"Qdrant Collection '{settings.COLLECTION_NAME}' already exists. Skipping Ingestion.")
            return
        
    except Exception:
        print(f"Collection '{settings.COLLECTION_NAME}' not found. Initializing one...")

        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(size=settings.VECTOR_SIZE, distance=Distance.COSINE)
        )
    
    print(f"Loading preprocessed data from {settings.CLEANED_DATA_PATH}...")
    df = pd.read_csv(settings.CLEANED_DATA_PATH)

    docs = convert_df_to_documents(df)

    print("Running local embeddings process and uploading to Qdrant...")
    QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name=settings.COLLECTION_NAME,
        batch_size=64,
        timeout=120
    )

    print("Success! All vectors are now securely stored in Qdrant Vector DB.")


if __name__ == "__main__":
    ingest_data()
