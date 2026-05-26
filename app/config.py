"""
app/config.py — centralised settings loaded from .env
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.1-70b-instruct:free"

    # MCP server
    mcp_server_path: str = "/path/to/server.py"
    mcp_python_bin: str = "python3"

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "oncology_platform"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "oncology_docs"

    # App
    app_env: str = "development"
    app_secret: str = "change_me"
    cors_origins: str = "http://localhost:3000"
    max_tool_iterations: int = 10
    stream_chunk_delay: float = 0.02

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def use_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def use_openrouter(self) -> bool:
        return bool(self.openrouter_api_key) and not self.use_groq


@lru_cache
def get_settings() -> Settings:
    return Settings()
