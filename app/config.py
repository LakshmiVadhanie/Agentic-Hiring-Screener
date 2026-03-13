"""Application configuration using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Agentic Hiring Screener"
    app_env: str = "development"
    log_level: str = "INFO"

    # Mistral
    mistral_api_key: str = ""

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "hiring_screener"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection: str = "resumes"

    # Embeddings
    embedding_model: str = "all-mpnet-base-v2"

    # Screening defaults
    default_top_k: int = 10
    default_threshold: float = 0.5
    similarity_weight: float = 0.4
    llm_weight: float = 0.6

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
