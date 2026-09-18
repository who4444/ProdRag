"""Repo indexing pipeline: walk -> chunk -> embed -> Qdrant (`code_chunks`).

Deterministic repo ids + re-index deletes old points first, so indexing the same
repo twice is idempotent and never leaves orphaned chunks.
"""

import hashlib
import logging
import time
from pathlib import Path

from ..config import settings
from ..core import db, vectorstore
from ..core.embeddings import text as text_emb
from .chunker import chunk_file
from .schemas import RepoStats
from .walker import walk_repo

logger = logging.getLogger(__name__)

EMBED_BATCH = 32


def repo_id_for(root: Path) -> str:
    """Deterministic id from the resolved root path -> stable across re-indexes."""
    return hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:12]


def _embed_text(payload: dict) -> str:
    header = f"{payload['path']}::{payload['symbol']}" if payload.get("symbol") else payload["path"]
    return f"{header}\n{payload['text']}"


async def index_repo(path: str, repo_id: str | None = None, name: str | None = None) -> RepoStats:
    root = Path(path).expanduser()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"repo path not found or not a directory: {path}")
    root = root.resolve()
    rid = repo_id or repo_id_for(root)
    t0 = time.perf_counter()

    await vectorstore.ensure_collections()
    await vectorstore.delete_code_by_repo(rid)

    files, skipped = walk_repo(root, settings.repo_max_files, settings.repo_max_file_bytes)
    payloads: list[dict] = []
    for f in files:
        for chunk in chunk_file(
            f, max_lines=settings.repo_chunk_max_lines, overlap=settings.repo_chunk_overlap_lines
        ):
            payload = chunk.model_dump()
            payload["repo_id"] = rid
            payloads.append(payload)

    upserted = 0
    for i in range(0, len(payloads), EMBED_BATCH):
        batch = payloads[i : i + EMBED_BATCH]
        vectors = await text_emb.embed_texts([_embed_text(p) for p in batch])
        upserted += await vectorstore.upsert_code(list(zip(batch, vectors)))

    stats = RepoStats(
        repo_id=rid,
        name=name or root.name,
        root=str(root),
        files=len(files),
        chunks=upserted,
        skipped=skipped,
        runtime_ms=int((time.perf_counter() - t0) * 1000),
    )
    try:
        await db.repo_upsert(
            {
                "id": rid,
                "name": stats.name,
                "root": stats.root,
                "files": stats.files,
                "chunks": stats.chunks,
                "status": "ready",
                "metadata": {"skipped": skipped, "runtime_ms": stats.runtime_ms},
            }
        )
    except Exception:
        logger.exception("repo_upsert failed for %s (index still searchable)", rid)
    logger.info(
        "repo indexed id=%s path=%s files=%d chunks=%d skipped=%d in %.2fs",
        rid,
        root,
        stats.files,
        stats.chunks,
        stats.skipped,
        stats.runtime_ms / 1000,
    )
    return stats
