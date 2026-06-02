from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    GROQ_API_KEY: str
    GROQ_MODEL_NAME: str

    QDRANT_API_KEY: str
    QDRANT_URL: str
    COLLECTION_NAME: str 

    MODEL_LOCAL_PATH: str 
    HUGGINGFACE_HUB_MODEL: str
    VECTOR_SIZE: int

    CLEANED_DATA_PATH: str

    HF_TOKEN: str
    class Config:
        env_file = ".env"


def get_settings():
    return Settings()
