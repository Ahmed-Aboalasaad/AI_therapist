import os
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document


from src.data_pipeline.data_preprocessing import run_preprocessing
from src.helpers.config import get_settings
from src.helpers.data_helpers import load_embeddings_model, convert_df_to_documents

settings = get_settings()



def ingest_data():
    print("Base Pipeline Core initialized.")

    run_preprocessing()

    embeddings = load_embeddings_model()

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

    print("Running local embeddings process on GPU and uploading to Qdrant...")
    QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name=settings.COLLECTION_NAME,
        batch_size=64,
        timeout=120
    )

    print(f"Success! All vectors are now securely stored in Qdrant Vector DB.")


if __name__ == "__main__":
    ingest_data()



