import json
from contextlib import asynccontextmanager
from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from openai import AsyncOpenAI

from . import storage, vectorstore
from .config import settings
from .embeddings import text as text_emb
from .embeddings import vision as vision_emb
from .query import answer
from .schemas import QueryRequest


async def require_token(authorization: str = Header(default="")):
    if authorization != f"Bearer {settings.api_token}":
        raise HTTPException(status_code=401, detail="invalid token")


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


@app.get("/health")
async def health():
    await app.state.redis.ping()
    return {"status": "ok"}


@app.post("/documents", dependencies=[Depends(require_token)])
async def upload_document(
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
):
    import uuid

    doc_id = str(uuid.uuid4())
    ext = Path(file.filename or "document.pdf").suffix or ".pdf"
    key = f"docs/{doc_id}/original{ext}"
    data = await file.read()
    await storage.put_bytes(key, data, file.content_type or "application/pdf")
    job = await app.state.redis.enqueue_job(
        "ingest_document", doc_id, key, json.loads(metadata)
    )
    return {"document_id": doc_id, "job_id": job.job_id}


@app.get("/documents/{doc_id}/status", dependencies=[Depends(require_token)])
async def document_status(doc_id: str):
    info = await app.state.redis.hgetall(f"doc:{doc_id}")
    if not info:
        raise HTTPException(status_code=404, detail="unknown document")
    return info


@app.get("/files/{object_key:path}", dependencies=[Depends(require_token)])
async def get_file(object_key: str):
    data = await storage.get_bytes(object_key)
    return Response(content=data, media_type="image/png")


@app.post("/query", dependencies=[Depends(require_token)])
async def query_endpoint(req: QueryRequest):
    vec = (await text_emb.embed_texts([req.question], query=True))[0]
    text_hits = await vectorstore.search_text(vec, req.k)

    image_hits: list[dict] = []
    if req.k_images > 0:
        try:
            q_vec = (await vision_emb.embed_texts([req.question]))[0]
            image_hits = await vectorstore.search_images(q_vec, req.k_images)
        except Exception:
            image_hits = []

    return StreamingResponse(
        answer(app.state.client, req.question, text_hits, image_hits),
        media_type="application/x-ndjson",
    )
