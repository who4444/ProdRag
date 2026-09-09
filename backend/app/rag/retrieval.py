"""Shared KB retrieval, used by both /query and the agent's kb_search tool.

Text retrieval is hybrid: dense (bge-m3) + sparse BM25 fused via RRF, then
optionally re-ranked by a cross-encoder (PRODRAG_RERANK_SERVICE_URL)."""

import logging
import time

from ..config import settings
from ..core import vectorstore
from ..core.embeddings import text as text_emb
from ..core.embeddings import vision as vision_emb
from ..core.rerank import rerank
from ..core.sparse import to_sparse

RERANK_TEXT_LIMIT = 1500  # chars fed to the cross-encoder per hit

logger = logging.getLogger(__name__)


async def search_kb(
    question: str, k: int = 4, k_images: int = 1
) -> tuple[list[dict], list[dict]]:
    t0 = time.perf_counter()
    first_k = max(settings.retrieval_hybrid_k, k * 3)
    logger.info("retrieve start k=%d k_images=%d hybrid_k=%d question=%r", k, k_images, first_k, question[:120])
    vec = (await text_emb.embed_texts([question], query=True))[0]
    sparse_q = to_sparse(question)
    text_hits = await vectorstore.search_text(
        vec, first_k, sparse_query=sparse_q, prefetch_k=first_k
    )
    logger.info("retrieve hybrid found %d text hits in %.2fs", len(text_hits), time.perf_counter() - t0)
    try:
        scores = await rerank(question, [h["text"][:RERANK_TEXT_LIMIT] for h in text_hits])
    except Exception:
        logger.warning("rerank failed for query=%r — falling back to RRF", question[:80], exc_info=True)
        scores = None
    if scores is not None:
        for h, s in zip(text_hits, scores):
            h["score"] = round(s, 3)
        text_hits.sort(key=lambda h: h["score"], reverse=True)
        text_hits = text_hits[:k]
        logger.info("retrieve reranked %d -> top %d in %.2fs", len(scores), len(text_hits), time.perf_counter() - t0)
    else:
        text_hits = text_hits[:k]
        if scores is None:
            # Check if rerank was enabled but failed — already logged above
            pass
    image_hits: list[dict] = []
    if k_images > 0:
        try:
            q_vec = (await vision_emb.embed_texts([question]))[0]
            image_hits = await vectorstore.search_images(q_vec, k_images)
            logger.info("retrieve %d image hits in %.2fs", len(image_hits), time.perf_counter() - t0)
        except Exception:
            logger.exception("image retrieval failed for question=%r", question[:120])
            image_hits = []
    logger.info(
        "retrieve complete %d text + %d image hits in %.2fs (top text: %s)",
        len(text_hits),
        len(image_hits),
        time.perf_counter() - t0,
        [f"{h['source']}#p{h['page']}:{h['score']}" for h in text_hits[:k]],
    )
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
            "image_url": f"/files/{h['object_key']}" if h.get("object_key") else None,
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
