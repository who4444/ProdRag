"""Shared KB retrieval, used by both /query and the agent's kb_search tool."""

from ..core import vectorstore
from ..core.embeddings import text as text_emb
from ..core.embeddings import vision as vision_emb


async def search_kb(
    question: str, k: int = 4, k_images: int = 1
) -> tuple[list[dict], list[dict]]:
    vec = (await text_emb.embed_texts([question], query=True))[0]
    text_hits = await vectorstore.search_text(vec, k)
    image_hits: list[dict] = []
    if k_images > 0:
        try:
            q_vec = (await vision_emb.embed_texts([question]))[0]
            image_hits = await vectorstore.search_images(q_vec, k_images)
        except Exception:
            image_hits = []
    return text_hits, image_hits


def source_items(text_hits: list[dict], image_hits: list[dict]) -> list[dict]:
    items = [
        {
            "kind": h.get("kind", "text"),
            "source": h["source"],
            "page": h["page"],
            "score": round(h["score"], 3),
            "content": h["text"][:200],
            "caption": h.get("caption", ""),
        }
        for h in text_hits
    ]
    items += [
        {
            "kind": "image",
            "source": h["source"],
            "page": h["page"],
            "score": round(h["score"], 3),
            "image_url": f"/files/{h['object_key']}",
            "caption": h.get("caption", ""),
        }
        for h in image_hits
    ]
    return items
