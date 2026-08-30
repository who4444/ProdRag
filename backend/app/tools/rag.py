"""Packaged RAG tool — thin wrapper over rag/retrieval.py.

Reused by /query, legacy /research agent, and R&D subagents. No duplication.
"""

from ..rag.retrieval import search_kb, source_items

__all__ = ["rag_search", "rag_source_items"]


async def rag_search(question: str, k: int = 4, k_images: int = 1) -> dict:
    """Run hybrid retrieval and return a dict with text + image hits."""
    text_hits, image_hits = await search_kb(question, k=k, k_images=k_images)
    return {"text": text_hits, "images": image_hits}


def rag_source_items(text_hits: list[dict], image_hits: list[dict]) -> list[dict]:
    return source_items(text_hits, image_hits)
