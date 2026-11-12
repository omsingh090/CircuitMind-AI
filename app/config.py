"""Application configuration settings for CircuitMind AI."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables and .env file."""

    # Project Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_raw_dir: Path = base_dir / "data" / "raw"
    data_processed_dir: Path = base_dir / "data" / "processed"

    # Google Gemini LLM Settings
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_temperature: float = 0.0
    gemini_max_tokens: int = 800

    @field_validator("gemini_model", mode="before")
    @classmethod
    def validate_gemini_model(cls, v: Optional[str]) -> str:
        """Fallback to gemini-2.5-flash if value is empty or unpopulated template placeholder."""
        if not v or "your_available_gemini_model" in v.lower() or "placeholder" in v.lower():
            return "gemini-2.5-flash"
        return v.strip()

    # Embeddings Configuration
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector Database Settings
    chroma_persist_directory: str = str(base_dir / "data" / "processed" / "chromadb")
    chroma_collection_name: str = "datasheets"

    # Ingestion & Chunking Defaults
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Application & Logging
    environment: str = "development"
    log_level: str = "INFO"

    # Server Defaults (for later milestones)
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_server_port: int = 8501

    # CORS Settings
    cors_origins: List[str] = Field(default=["*"], alias="CORS_ORIGINS")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def validate_cors_origins(cls, v: Any) -> List[str]:
        """Parse comma-separated string or list into list of allowed CORS origins."""
        if isinstance(v, str):
            origins = [o.strip() for o in v.split(",") if o.strip()]
            return origins if origins else ["*"]
        if isinstance(v, list):
            return [str(o).strip() for o in v if str(o).strip()]
        return ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of the application settings."""
    return Settings()

