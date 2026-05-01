from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str = ""
    anthropic_api_key: str = ""
    wandb_api_key: str = ""

    ollama_base_url: str = "http://localhost:11434"
    ollama_fallback_model: str = "llama3.2"

    sqlite_path: str = "./arbitration_audit.db"

    disagreement_threshold: int = 3

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
