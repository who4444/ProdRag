"""Subagent — one direction: rag_search + synthesis.

ponytail: plain async function, not a framework. Called via asyncio.gather by the orchestrator.
"""

import logging

from ..config import settings
from ..tools.rag import rag_search
from .schemas import Direction, ResearchSummary

logger = logging.getLogger(__name__)


def _format_hits(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        text = h.get("text", "")[:800]
        parts.append(f"[{i}] (page {h.get('page')}, score {h.get('score'):.2f}) {text}")
    return "\n\n".join(parts) or "No results."


async def run_direction(client, direction: Direction, k: int = 4, k_images: int = 1) -> ResearchSummary:
    """Execute one research direction: retrieve + synthesize."""
    from ..tools.rag import rag_source_items

    result = await rag_search(direction.question, k=k, k_images=k_images)
    text_hits = result["text"]
    image_hits = result["images"]
    sources = rag_source_items(text_hits, image_hits)

    context = _format_hits(text_hits)
    # figures as refs only (text-only model)
    if image_hits:
        context += "\n\nFigures:\n" + "\n".join(
            f"[fig {j+1}] page {img.get('page')} object_key={img.get('object_key')}" for j, img in enumerate(image_hits)
        )

    prompt = (
        f"Research direction: {direction.id} — {direction.question}\n"
        f"Rationale: {direction.rationale}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        "Task: summarize the key findings for this direction in 3-6 sentences, "
        "citing sources as [1], [2] where relevant. Then list any gaps. "
        "Be concise and grounded — never invent facts not in the evidence."
    )

    try:
        resp = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "user", "content": prompt}],
        )
        findings = (resp.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("subagent %s synthesis failed", direction.id)
        findings = _format_hits(text_hits)[:1200] or "Synthesis failed — raw evidence above."

    # confidence heuristic: grounded hit count
    if len(text_hits) >= 3:
        confidence = "high"
    elif len(text_hits) >= 1:
        confidence = "med"
    else:
        confidence = "low"

    gaps: list[str] = []
    if confidence == "low":
        gaps.append(f"Low evidence for {direction.id}: only {len(text_hits)} hits")

    return ResearchSummary(
        direction_id=direction.id,
        findings=findings,
        sources=sources,
        confidence=confidence,
        gaps=gaps,
    )
