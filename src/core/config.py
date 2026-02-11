from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str
    API_V1_STR: str
    BASE_MODEL_NAME: str
    ADAPTER_PATH: Optional[str] = None
    HOST: str
    PORT: int
    LOG_LEVEL: str

    class Config:
        env_file = "config/.env"

settings = Settings()