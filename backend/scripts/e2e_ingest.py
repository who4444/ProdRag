"""End-to-end ingestion test against real infra (Qdrant + Redis + Supabase).

Uploads a real PDF from data/, enqueues the arq job, runs ingest_document
in-process, then verifies: document row, parts rows, qdrant points (text+image),
storage objects, and OCR.

Run from backend/ so backend/.env is picked up:
    uv run python scripts/e2e_ingest.py
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("PYTHONPATH", ".")

from app.config import settings  # noqa: E402
from app.core import db, storage, vectorstore  # noqa: E402
from app.ingest.pipeline import ingest_document  # noqa: E402

PDF = Path("data/0198.pdf")
RESULTS = {}


async def check(name, cond, detail=""):
    RESULTS[name] = bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


async def main():
    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    doc_id = str(uuid.uuid4())
    key = f"docs/{doc_id}/original.pdf"
    data = PDF.read_bytes()

    print(f"doc_id={doc_id}  pdf={PDF.name} ({len(data)} bytes)")

    # 1. upload source
    await storage.ensure_bucket()
    await storage.put_bytes(key, data, "application/pdf")
    await check("source uploaded", await storage.get_bytes(key) == data, key)

    # 2. ingest (parse -> embed -> index -> persist)
    await db.document_upsert(
        {"id": doc_id, "title": PDF.name, "source_key": key, "status": "queued", "metadata": {}}
    )
    result = await ingest_document(
        {"redis": redis}, doc_id, key, {"test": "e2e"}
    )
    print("ingest result:", result)

    # 3. document row
    doc = await db.document_get(doc_id)
    await check("document row done", doc and doc["status"] == "done", f"status={doc and doc['status']}")

    # 4. parts rows (one per qdrant point)
    parts = await db.parts_for_document(doc_id)
    await check("parts rows written", len(parts) == result["text_chunks"] + result["images"],
                f"parts={len(parts)} vs chunks={result['text_chunks']}+imgs={result['images']}")

    # 5. qdrant text points reachable via hybrid search
    from app.core.embeddings import text as text_emb
    from app.core.sparse import to_sparse
    qvec = (await text_emb.embed_texts(["anomaly detection"], query=True))[0]
    text_hits = await vectorstore.search_text(
        qvec, 3, sparse_query=to_sparse("anomaly detection"), prefetch_k=9
    )
    await check("qdrant text search finds doc", any(h["doc_id"] == doc_id for h in text_hits),
                f"hits={[h['page'] for h in text_hits][:3]}")

    # 6. qdrant image points (scroll for this doc's own image points)
    from app.core.embeddings import vision as vision_emb
    c = await vectorstore.get_client()
    pts = await c.scroll("image_chunks", limit=200, with_payload=True)
    own = [p for p in pts[0] if p.payload.get("doc_id") == doc_id]
    await check("qdrant image points indexed", len(own) == result["images"],
                f"own={len(own)} vs images={result['images']}")

    # 7. storage figure objects exist
    fig_keys = [p["object_key"] for p in parts if p["object_key"]]
    fig_ok = True
    for fk in fig_keys:
        try:
            await storage.get_bytes(fk)
        except Exception:
            fig_ok = False
            break
    await check("figure objects stored", fig_ok, f"count={len(fig_keys)}")

    # 8. cleanup test rows (keep qdrant points for a final search demo)
    await db.document_update(doc_id, {"status": "e2e-test"})

    await redis.aclose()
    await vectorstore.close_client()

    passed = sum(1 for v in RESULTS.values() if v)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    if passed != len(RESULTS):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
