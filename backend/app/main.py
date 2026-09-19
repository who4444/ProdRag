from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from .api import router
from .config import settings
from .core import storage, vectorstore
from .runs import checkpoint as run_checkpoint
from .runs.graph import reset_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_checkpoint.init_checkpointer()
    reset_graph()
    if not settings.offline:
        await storage.ensure_bucket()
    await vectorstore.ensure_collections()
    app.state.redis = None
    if not settings.offline:
        app.state.redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    app.state.client = AsyncOpenAI(
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url or None,
    )
    yield
    if app.state.redis is not None:
        await app.state.redis.aclose()
    await vectorstore.close_client()
    await run_checkpoint.close_checkpointer()


app = FastAPI(title="ProdRag", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/health")
async def health():
    if settings.offline:
        return {"status": "ok", "offline": True}
    await app.state.redis.ping()
    return {"status": "ok"}
