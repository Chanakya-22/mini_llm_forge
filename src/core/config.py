"""Production-grade configuration management for the backend service.

This module centralizes environment-driven settings for the application using
Pydantic V2 and ``pydantic-settings``. It is designed to be strict for known
backend variables while safely ignoring unrelated environment entries such as
frontend-specific keys.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and the .env file.

    The schema is intentionally explicit for backend configuration while using
    ``extra="ignore"`` so unrelated variables in the shared environment file do
    not trigger validation failures.
    """

    model_config = SettingsConfigDict(
        env_file="config/.env",
        extra="ignore",
        case_sensitive=False,
        env_file_encoding="utf-8",
    )

    project_name: str = Field(
        default="Mini-LLM-Forge",
        description="The public name of the application.",
    )
    api_v1_str: str = Field(
        default="/api/v1",
        description="The API version prefix used by the backend router.",
    )
    base_model_name: str = Field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="The Hugging Face identifier for the base model.",
    )
    adapter_path: Optional[str] = Field(
        default=None,
        description="Optional filesystem path to a LoRA adapter directory.",
    )
    host: str = Field(
        default="0.0.0.0",
        description="The host interface on which the API server should listen.",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="The TCP port on which the API server should listen.",
    )
    log_level: str = Field(
        default="INFO",
        description="The logging verbosity level used by the application.",
    )
    api_url: str = Field(
        default="http://localhost:8000",
        description="The backend API base URL used by downstream clients.",
    )


settings = Settings()