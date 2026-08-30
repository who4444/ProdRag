"""arq job that parses a PDF, embeds text + figures, and indexes into Qdrant."""

import logging
import time
from contextlib import contextmanager

from ..config import settings
from ..core import db, storage, vectorstore
from ..core.embeddings import text as text_emb
from ..core.embeddings import vision as vision_emb
from ..core.ocr import extract_text as ocr_extract
from .chunking import chunk_text
from .parser.pdf import parse_pdf

logger = logging.getLogger(__name__)


@contextmanager
def _stage(name: str, **fields):
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception("ingest stage %s failed for %s", name, fields)
        raise
    logger.info("ingest %s done in %.2fs (%s)", name, time.perf_counter() - t0, fields)


async def ingest_document(ctx: dict, doc_id: str, object_key: str, metadata: dict | None = None):
    metadata = metadata or {}
    redis = ctx["redis"]
    title = object_key.rsplit("/", 1)[-1] or object_key
    t_start = time.perf_counter()
    logger.info("ingest start doc_id=%s source=%s", doc_id, object_key)
    await db.document_upsert(
        {
            "id": doc_id,
            "title": title,
            "source_key": object_key,
            "status": "processing",
            "metadata": metadata,
        }
    )
    try:
        with _stage("cleanup", doc_id=doc_id):
            parts = await db.parts_for_document(doc_id)
            if parts:
                await vectorstore.delete_points([p["id"] for p in parts])
                for key in (p["object_key"] for p in parts if p.get("object_key")):
                    await storage.delete_object(key)
                await db.part_delete_for_document(doc_id)

        with _stage("ensure", doc_id=doc_id):
            await storage.ensure_bucket()
            await vectorstore.ensure_collections()

        with _stage("download", doc_id=doc_id, source=object_key):
            data = await storage.get_bytes(object_key)

        with _stage("parse", doc_id=doc_id):
            document = parse_pdf(data, dpi=settings.page_dpi, min_figure_area=settings.min_figure_area)
        n_pages = len(document.pages)
        n_figs = sum(len(p.figures) for p in document.pages)
        n_tables = sum(len(p.tables) for p in document.pages)
        logger.info("ingest parsed doc_id=%s pages=%d figures=%d tables=%d", doc_id, n_pages, n_figs, n_tables)

        # --- text + table chunks ---
        text_payloads: list[dict] = []
        texts: list[str] = []
        with _stage("chunk", doc_id=doc_id):
            for page in document.pages:
                if page.text.strip():
                    for ci, chunk in enumerate(chunk_text(page.text)):
                        texts.append(chunk)
                        text_payloads.append(
                            {
                                "doc_id": doc_id,
                                "source": object_key,
                                "page": page.number,
                                "chunk_idx": ci,
                                "kind": "text",
                                "metadata": metadata,
                                "text": chunk,
                            }
                        )
                for table in page.tables:
                    texts.append(table.markdown)
                    text_payloads.append(
                        {
                            "doc_id": doc_id,
                            "source": object_key,
                            "page": page.number,
                            "chunk_idx": f"table_{table.index}",
                            "kind": "table",
                            "caption": table.caption.text if table.caption else "",
                            "block_order": table.order,
                            "metadata": metadata,
                            "text": table.markdown,
                        }
                    )
        with _stage("embed_text", doc_id=doc_id, n=len(texts)):
            embeddings = await text_emb.embed_texts(texts)
        with _stage("upsert_text", doc_id=doc_id, n=len(text_payloads)):
            n_text = await vectorstore.upsert_text(list(zip(text_payloads, embeddings)))
        logger.info("ingest text indexed doc_id=%s chunks=%d", doc_id, n_text)

        # --- figures / images ---
        image_payloads: list[dict] = []
        blobs: list[bytes] = []
        ocr_texts: list[str] = []
        ocr_payloads: list[dict] = []
        with _stage("figures", doc_id=doc_id, n_figures=n_figs):
            for page in document.pages:
                for fig in page.figures:
                    key = f"docs/{doc_id}/figures/{page.number}_{fig.index}.png"
                    await storage.put_bytes(key, fig.data, "image/png")
                    ocr = ocr_extract(fig.data)
                    image_payloads.append(
                        {
                            "doc_id": doc_id,
                            "source": object_key,
                            "page": page.number,
                            "image": fig.index,
                            "object_key": key,
                            "caption": fig.caption.text if fig.caption else "",
                            "ocr_text": ocr,
                            "block_order": fig.order,
                            "metadata": metadata,
                        }
                    )
                    blobs.append(fig.data)
                    if ocr:
                        ocr_texts.append(ocr)
                        ocr_payloads.append(
                            {
                                "doc_id": doc_id,
                                "source": object_key,
                                "page": page.number,
                                "chunk_idx": f"figure_{page.number}_{fig.index}",
                                "kind": "figure_ocr",
                                "object_key": key,
                                "caption": fig.caption.text if fig.caption else "",
                                "metadata": metadata,
                                "text": ocr,
                            }
                        )
        with _stage("embed_image", doc_id=doc_id, n=len(blobs)):
            image_vectors = await vision_emb.embed_images(blobs)
        with _stage("upsert_image", doc_id=doc_id, n=len(image_payloads)):
            n_images = await vectorstore.upsert_image(list(zip(image_payloads, image_vectors)))
        logger.info("ingest images indexed doc_id=%s images=%d ocr=%d", doc_id, n_images, len(ocr_texts))

        # OCR text indexed as searchable chunks so text queries can find figures.
        if ocr_texts:
            with _stage("embed_ocr", doc_id=doc_id, n=len(ocr_texts)):
                ocr_embeddings = await text_emb.embed_texts(ocr_texts)
            with _stage("upsert_ocr", doc_id=doc_id, n=len(ocr_payloads)):
                n_ocr = await vectorstore.upsert_text(list(zip(ocr_payloads, ocr_embeddings)))
            n_text += n_ocr

        # Record index entries (one per qdrant point) for delete-doc / listings.
        parts = [
            {
                "id": await vectorstore.point_id("text", p),
                "document_id": doc_id,
                "kind": p["kind"],
                "page": p["page"],
                "block_order": p.get("block_order"),
                "caption": p.get("caption", ""),
                "ocr_text": p.get("ocr_text", ""),
                "object_key": p.get("object_key", ""),
            }
            for p in text_payloads + ocr_payloads
        ]
        parts += [
            {
                "id": await vectorstore.point_id("image", p),
                "document_id": doc_id,
                "kind": "image",
                "page": p["page"],
                "block_order": p.get("block_order"),
                "caption": p.get("caption", ""),
                "ocr_text": p.get("ocr_text", ""),
                "object_key": p.get("object_key", ""),
            }
            for p in image_payloads
        ]
        await db.part_insert_many(parts)
        logger.info("ingest parts recorded doc_id=%s parts=%d", doc_id, len(parts))
        await db.document_update(
            doc_id,
            {"status": "done", "text_chunks": n_text, "images": n_images, "error": None},
        )
        await redis.hset(
            f"doc:{doc_id}",
            mapping={"status": "done", "text_chunks": n_text, "images": n_images},
        )
        logger.info("ingest complete doc_id=%s text_chunks=%d images=%d total=%.2fs", doc_id, n_text, n_images, time.perf_counter() - t_start)
        return {"text_chunks": n_text, "images": n_images}
    except Exception as exc:
        logger.exception("ingest failed doc_id=%s source=%s", doc_id, object_key)
        await db.document_update(doc_id, {"status": "error", "error": str(exc)})
        await redis.hset(f"doc:{doc_id}", mapping={"status": "error", "error": str(exc)})
        raise
