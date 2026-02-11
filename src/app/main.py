from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.core.config import settings
from src.core.logger import logger
from src.engine.generator import engine
from src.app.api.v1 import chat

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server Starting...")
    engine.load_model()
    yield
    logger.info("Server Shutting Down...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

app.include_router(chat.router, prefix=settings.API_V1_STR)

@app.get("/health")
def health():
    return {"status": "ok"}