from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI
from openai import AsyncOpenAI

from .api import router
from .config import settings
from .core import storage, vectorstore


@asynccontextmanager
async def lifespan(app: FastAPI):
    await storage.ensure_bucket()
    await vectorstore.ensure_collections()
    app.state.redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    app.state.client = AsyncOpenAI(
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url or None,
    )
    yield
    await app.state.redis.aclose()
    await vectorstore.close_client()


app = FastAPI(title="ProdRag", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health():
    await app.state.redis.ping()
    return {"status": "ok"}
