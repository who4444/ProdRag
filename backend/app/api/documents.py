import hashlib
import json
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..core import db, storage, vectorstore

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
    content_hash = hashlib.sha256(data).hexdigest()
    existing = await db.document_by_hash(content_hash)
    if existing:
        return {"document_id": existing["id"], "job_id": None, "duplicate": True}
    await storage.put_bytes(key, data, file.content_type or "application/pdf")
    await db.document_upsert(
        {
            "id": doc_id,
            "title": file.filename or key,
            "source_key": key,
            "content_hash": content_hash,
            "status": "queued",
            "metadata": json.loads(metadata),
        }
    )
    job = await request.app.state.redis.enqueue_job(
        "ingest_document", doc_id, key, json.loads(metadata)
    )
    return {"document_id": doc_id, "job_id": job.job_id}


@router.post("/documents/{doc_id}/reingest")
async def reingest_document(request: Request, doc_id: str):
    doc = await db.document_get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="unknown document")
    job = await request.app.state.redis.enqueue_job(
        "ingest_document", doc_id, doc["source_key"], doc.get("metadata") or {}
    )
    return {"document_id": doc_id, "job_id": job.job_id}


@router.get("/documents/{doc_id}/status")
async def document_status(doc_id: str):
    info = await db.document_get(doc_id)
    if not info:
        raise HTTPException(status_code=404, detail="unknown document")
    return info


@router.get("/documents")
async def document_list(status: str | None = None):
    return await db.document_list(status)


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    doc = await db.document_get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="unknown document")
    parts = await db.parts_for_document(doc_id)
    if parts:
        await vectorstore.delete_points([p["id"] for p in parts])
    # Drop S3 objects: the source PDF + every stored figure.
    keys = [doc["source_key"]] + [p["object_key"] for p in parts if p.get("object_key")]
    for key in keys:
        await storage.delete_object(key)
    await db.document_delete(doc_id)
    return {"deleted": doc_id}
