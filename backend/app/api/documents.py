import json
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..core import storage

router = APIRouter()


@router.post("/documents")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
):
    import uuid

    doc_id = str(uuid.uuid4())
    ext = Path(file.filename or "document.pdf").suffix or ".pdf"
    key = f"docs/{doc_id}/original{ext}"
    data = await file.read()
    await storage.put_bytes(key, data, file.content_type or "application/pdf")
    job = await request.app.state.redis.enqueue_job(
        "ingest_document", doc_id, key, json.loads(metadata)
    )
    return {"document_id": doc_id, "job_id": job.job_id}


@router.get("/documents/{doc_id}/status")
async def document_status(request: Request, doc_id: str):
    info = await request.app.state.redis.hgetall(f"doc:{doc_id}")
    if not info:
        raise HTTPException(status_code=404, detail="unknown document")
    return info
