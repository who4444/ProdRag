import base64
import json

from openai import AsyncOpenAI

from ..core import storage
from ..config import settings
from .retrieval import source_items


async def answer(
    client: AsyncOpenAI,
    question: str,
    text_hits: list[dict],
    image_hits: list[dict],
):
    """Yields NDJSON: a sources event (text + image refs), then text deltas."""
    yield json.dumps({"type": "sources", "items": source_items(text_hits, image_hits)}) + "\n"

    context = "\n\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(text_hits))
    if settings.chat_supports_images:
        user_content: list[dict] = [
            {
                "type": "text",
                "text": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ]
        for h in image_hits[: settings.max_images_in_context]:
            blob = await storage.get_bytes(h["object_key"])
            b64 = base64.b64encode(blob).decode()
            user_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )
    else:
        user_content = f"Context:\n{context}\n\nQuestion: {question}"

    system = (
        "You are a RAG assistant. Answer only from the provided context. "
        "Cite text sources as [1], [2], ... If the context does not answer the "
        "question, say so."
    )
    if settings.chat_supports_images:
        system += " Figures are attached as images — reference them when relevant."

    stream = await client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        stream=True,
    )
    async for part in stream:
        delta = part.choices[0].delta.content
        if delta:
            yield json.dumps({"type": "text", "delta": delta}) + "\n"
