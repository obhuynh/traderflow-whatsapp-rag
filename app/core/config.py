# The pydantic_settings package is now used for loading from .env files
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # This replaces the old 'class Config:' to specify the .env file
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # App
    PROJECT_NAME: str = "TraderFlow WhatsApp RAG"
    API_V1_STR: str = "/api/v1"

    # Twilio
    TWILIO_ACCOUNT_SID: str
    TWILIO_AUTH_TOKEN: str
    TWILIO_PHONE_NUMBER: str

    # Hugging Face & Ollama
    HF_TOKEN: str
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    LLM_MODEL_ID: str

    # ChromaDB
    CHROMA_HOST: str = "chroma"
    CHROMA_PORT: int = 8002

# This line remains the same
settings = Settings()