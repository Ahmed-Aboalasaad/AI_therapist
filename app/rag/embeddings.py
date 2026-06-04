import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from app.core.config import get_settings
from app.repositories.qdrant_repository import QdrantRepository

settings = get_settings()
_embedding_model = None

def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    Loads embedding model offline if found in local assets path, otherwise downloads it.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check default path, fallback assets path, and legacy models path
    model_paths_to_check = [
        settings.MODEL_LOCAL_PATH,
        os.path.join("app", "assets", "embeddings", "all-MiniLM-L6-v2"),
        os.path.join("app", "models", "embeddings", "all-MiniLM-L6-v2")
    ]
    
    selected_path = settings.MODEL_LOCAL_PATH
    found_local = False
    
    for path in model_paths_to_check:
        local_weight_file = os.path.join(path, "model.safetensors")
        if os.path.exists(local_weight_file):
            selected_path = path
            found_local = True
            break
            
    if found_local:
        print(f"Found local Embedding Model offline at: '{selected_path}'. Loading directly...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=selected_path,
            model_kwargs={'device': device} 
        )
    else:
        print(f"Local model NOT found at expected paths. Downloading '{settings.HUGGINGFACE_HUB_MODEL}' from Hugging Face Hub...")
        os.makedirs(settings.MODEL_LOCAL_PATH, exist_ok=True)
        embedding_model = HuggingFaceEmbeddings(
            model_name=settings.HUGGINGFACE_HUB_MODEL,
            cache_folder=os.path.dirname(settings.MODEL_LOCAL_PATH),
            model_kwargs={'device': device} 
        )
        print("Model downloaded successfully and cached locally!")

    _embedding_model = embedding_model
    return _embedding_model

def get_vector_store(qdrant_repo: QdrantRepository) -> QdrantVectorStore:
    """
    Returns an instance of LangChain's QdrantVectorStore initialized with Qdrant client.
    """
    embeddings = load_embedding_model()
    return QdrantVectorStore(
        client=qdrant_repo.client,
        collection_name=settings.COLLECTION_NAME,
        embedding=embeddings
    )
