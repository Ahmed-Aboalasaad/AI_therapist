import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import pandas as pd
from src.helpers.config import get_settings
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



def load_embeddings_model() -> HuggingFaceEmbeddings:

    MODEL_LOCAL_PATH = settings.MODEL_LOCAL_PATH
    HUGGINGFACE_HUB_MODEL = settings.HUGGINGFACE_HUB_MODEL

    local_weight_file = os.path.join(MODEL_LOCAL_PATH, "model.safetensors")
    
    if os.path.exists(local_weight_file):
        print(f"Found local Embedding Model offline at: '{MODEL_LOCAL_PATH}'. Loading directly...")
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_LOCAL_PATH,
            model_kwargs={'device': 'cuda'} 
        )

    else:
        print(f"Local model NOT found at '{MODEL_LOCAL_PATH}'.")
        print(f"Downloading '{HUGGINGFACE_HUB_MODEL}' from Hugging Face Hub...")

        os.makedirs(MODEL_LOCAL_PATH, exist_ok=True)
        
        embeddings = HuggingFaceEmbeddings(
            model_name=HUGGINGFACE_HUB_MODEL,
            cache_folder=os.path.dirname(MODEL_LOCAL_PATH),
            model_kwargs={'device': 'cuda'} 
        )

        print(f"Model downloaded successfully and cached locally!")


    return embeddings
