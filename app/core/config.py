from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the root workspace directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    GROQ_API_KEY: str
    GROQ_MODEL_NAME: str

    QDRANT_API_KEY: str
    QDRANT_URL: str
    COLLECTION_NAME: str 

    MODEL_LOCAL_PATH: str = "app/assets/embeddings/all-MiniLM-L6-v2"
    HUGGINGFACE_HUB_MODEL: str
    VECTOR_SIZE: int

    CLEANED_DATA_PATH: str

    HF_TOKEN: str

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()
