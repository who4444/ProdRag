"""arq job that parses a PDF, embeds text + figures, and indexes into Qdrant."""

from .config import settings
from . import storage, vectorstore
from .chunking import chunk_text
from .embeddings import text as text_emb
from .embeddings import vision as vision_emb
from .parser.pdf import parse_pdf


async def ingest_document(ctx: dict, doc_id: str, object_key: str, metadata: dict | None = None):
    metadata = metadata or {}
    redis = ctx["redis"]
    await redis.hset(f"doc:{doc_id}", mapping={"status": "processing", "error": ""})
    try:
        await storage.ensure_bucket()
        await vectorstore.ensure_collections()

        data = await storage.get_bytes(object_key)
        pages = parse_pdf(data, dpi=settings.page_dpi, min_figure_area=settings.min_figure_area)

        # --- text chunks ---
        text_payloads: list[dict] = []
        texts: list[str] = []
        for page in pages:
            if not page.text.strip():
                continue
            for ci, chunk in enumerate(chunk_text(page.text)):
                texts.append(chunk)
                text_payloads.append(
                    {
                        "doc_id": doc_id,
                        "source": object_key,
                        "page": page.number,
                        "chunk_idx": ci,
                        "metadata": metadata,
                        "text": chunk,
                    }
                )
        embeddings = await text_emb.embed_texts(texts)
        n_text = await vectorstore.upsert_text(list(zip(text_payloads, embeddings)))

        # --- figures / images ---
        image_payloads: list[dict] = []
        blobs: list[bytes] = []
        for page in pages:
            for fig in page.figures:
                key = f"docs/{doc_id}/figures/{page.number}_{fig.index}.png"
                await storage.put_bytes(key, fig.data, "image/png")
                image_payloads.append(
                    {
                        "doc_id": doc_id,
                        "source": object_key,
                        "page": page.number,
                        "image": fig.index,
                        "object_key": key,
                        "metadata": metadata,
                    }
                )
                blobs.append(fig.data)
        image_vectors = await vision_emb.embed_images(blobs)
        n_images = await vectorstore.upsert_image(list(zip(image_payloads, image_vectors)))

        await redis.hset(
            f"doc:{doc_id}",
            mapping={"status": "done", "text_chunks": n_text, "images": n_images},
        )
        return {"text_chunks": n_text, "images": n_images}
    except Exception as exc:
        await redis.hset(f"doc:{doc_id}", mapping={"status": "error", "error": str(exc)})
        raise
