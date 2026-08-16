"""Memory: conversation (Redis) + episodic (Qdrant `episodes`)."""

import json
import time
import uuid

from .core import vectorstore
from .config import settings
from .core.embeddings import text as text_emb

CONV_MAX = 20


async def conv_add(redis, session_id: str | None, role: str, content: str) -> str:
    session_id = session_id or str(uuid.uuid4())
    key = f"conv:{session_id}"
    await redis.rpush(key, json.dumps({"role": role, "content": content}))
    await redis.ltrim(key, -CONV_MAX, -1)
    await redis.expire(key, settings.memory_ttl_s)
    return session_id


async def conv_history(redis, session_id: str | None, n: int = CONV_MAX) -> list[dict]:
    if not session_id:
        return []
    raw = await redis.lrange(f"conv:{session_id}", -n, -1)
    return [json.loads(x) for x in raw]


async def remember_search(question: str, k: int | None = None) -> list[dict]:
    k = k or settings.memory_top_k
    vec = (await text_emb.embed_texts([question], query=True))[0]
    return await vectorstore.search_episodes(vec, k)


async def remember_save(
    session_id: str | None, question: str, answer: str, sources: list[dict]
) -> str | None:
    payload = {
        "session_id": session_id,
        "created_at": time.time(),
        "question": question,
        "answer": answer,
        "sources": sources[:10],
    }
    vec = (await text_emb.embed_texts([f"Q: {question}\nA: {answer}"]))[0]
    return await vectorstore.upsert_episodes([(payload, vec)])
