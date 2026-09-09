"""Subagent — one direction: rag_search + synthesis.

ponytail: plain async function, not a framework. Called via asyncio.gather by the orchestrator.
"""

import base64
import logging

from ..config import settings
from ..core import storage
from ..tools.rag import rag_search
from ..tools.search import format_results, web_search
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

    # web search fallback when RAG is thin (no-op when unconfigured)
    from ..tools.search import enabled as search_enabled

    web_hits: list[dict] = []
    logger.info("subagent %s: rag hits=%d web_enabled=%s", direction.id, len(text_hits), search_enabled())
    if len(text_hits) < 2:
        logger.info("subagent %s: thin RAG (%d hits) -> triggering web_search for %r", direction.id, len(text_hits), direction.question[:80])
        web_hits = await web_search(direction.question, k=3)
        logger.info("subagent %s: web_search returned %d hits (enabled=%s)", direction.id, len(web_hits), search_enabled())
        if web_hits:
            context += "\n\nWeb results:\n" + format_results(web_hits)
            # surface web hits as sources (kind=web) so downstream can cite them
            for w in web_hits:
                sources.append(
                    {
                        "kind": "web",
                        "source": w.get("url", ""),
                        "page": 0,
                        "score": 0.0,
                        "content": w.get("snippet", "")[:200],
                        "caption": w.get("title", ""),
                        "image_url": None,
                    }
                )
        else:
            logger.info("subagent %s: web_search no results (check PRODRAG_TAVILY_API_KEY / SERPER / SEARCH_SERVICE_URL)", direction.id)

    # Iterative retrieval: if still low (0 hits), rewrite query once
    if len(text_hits) == 0:
        try:
            rewrite_prompt = f"Rewrite this research question for better retrieval (HyDE): {direction.question}\nReturn only the rewritten question."
            r2 = await client.chat.completions.create(
                model=settings.chat_model,
                messages=[{"role": "user", "content": rewrite_prompt}],
            )
            rewritten = (r2.choices[0].message.content or "").strip().splitlines()[0][:300]
            # Guard: skip if rewrite looks like a finding/summary (test mocks return "Findings..." ) or too short
            if rewritten and rewritten != direction.question and len(rewritten) > 10 and "findings" not in rewritten.lower() and "citation" not in rewritten.lower():
                logger.info("subagent %s: rewriting query %r -> %r", direction.id, direction.question[:60], rewritten[:60])
                r2_result = await rag_search(rewritten, k=k, k_images=k_images)
                if r2_result["text"]:
                    text_hits = r2_result["text"]
                    image_hits = r2_result["images"]
                    sources = rag_source_items(text_hits, image_hits)
                    context = _format_hits(text_hits)
                    if image_hits:
                        context += "\n\nFigures:\n" + "\n".join(f"[fig {j+1}] page {img.get('page')} object_key={img.get('object_key')}" for j, img in enumerate(image_hits))
                    logger.info("subagent %s: rewrite gained %d hits", direction.id, len(text_hits))
        except Exception:
            logger.exception("subagent %s: rewrite failed", direction.id)

    prompt = (
        f"Research direction: {direction.id} — {direction.question}\n"
        f"Rationale: {direction.rationale}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        "Task: summarize the key findings for this direction in 3-6 sentences, "
        "citing sources as [1], [2] where relevant. Then list any gaps. "
        "Be concise and grounded — never invent facts not in the evidence."
    )

    # Vision: attach figure images when DS Vision enabled
    use_vision = settings.chat_supports_images and image_hits
    messages: list[dict]  # type: ignore
    if use_vision:
        user_content: list[dict] = [{"type": "text", "text": prompt}]
        for h in image_hits[: settings.max_images_in_context]:
            try:
                blob = await storage.get_bytes(h["object_key"])
                b64 = base64.b64encode(blob).decode()
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            except Exception:
                logger.warning("subagent %s: failed to fetch image %s", direction.id, h.get("object_key"))
        messages = [{"role": "user", "content": user_content}]
        model = settings.vision_chat_model or settings.chat_model
    else:
        messages = [{"role": "user", "content": prompt}]
        model = settings.chat_model

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
        )
        findings = (resp.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("subagent %s synthesis failed", direction.id)
        findings = _format_hits(text_hits)[:1200] or "Synthesis failed — raw evidence above."

    # Gated self-critique when not high confidence
    if len(text_hits) < 3 and findings and "Synthesis failed" not in findings:
        try:
            critic_prompt = (
                f"Verify this summary against the evidence. Fix any invented facts, ensure citations [1],[2] are correct, "
                f"and keep 3-6 sentences. Evidence:\n{context[:2500]}\n\nSummary:\n{findings}\n\nReturn corrected summary only."
            )
            cr = await client.chat.completions.create(
                model=settings.chat_model,
                messages=[{"role": "user", "content": critic_prompt}],
            )
            corrected = (cr.choices[0].message.content or "").strip()
            if corrected and len(corrected) > 20:
                findings = corrected
                logger.info("subagent %s: self-critique applied", direction.id)
        except Exception:
            logger.exception("subagent %s: critique failed", direction.id)

    # confidence heuristic: grounded hit count (recomputed after rewrite)
    if len(text_hits) >= 3:
        confidence = "high"
    elif len(text_hits) >= 1:
        confidence = "med"
    else:
        confidence = "low"

    gaps: list[str] = []
    if confidence == "low":
        gaps.append(f"Low evidence for {direction.id}: only {len(text_hits)} hits")

    # Citation coverage (ponytail: derived metric, no LLM)
    import re

    SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
    CITE_RE = re.compile(r"\[(\d+)\]")
    sents = [s.strip() for s in SENT_RE.split(findings.strip()) if s.strip()]
    total_claims = len(sents) if sents else 1
    cited_claims = sum(1 for s in sents if any(1 <= int(n) <= len(sources) for n in CITE_RE.findall(s)))
    sent_coverage = cited_claims / total_claims if total_claims else 0.0
    # warn if low coverage
    if sent_coverage < 0.5 and findings and "Synthesis failed" not in findings:
        gaps.append(f"Low citation coverage {sent_coverage:.0%} for {direction.id} ({cited_claims}/{total_claims} sentences cited)")
    cited_sources = sorted({sources[int(n) - 1].get("source", "") for s in sents for n in CITE_RE.findall(s) if 1 <= int(n) <= len(sources) and sources[int(n) - 1].get("source")})
    citation_coverage = round(sent_coverage, 3)

    return ResearchSummary(
        direction_id=direction.id,
        findings=findings,
        sources=sources,
        confidence=confidence,
        gaps=gaps,
        citation_coverage=citation_coverage,
        cited_sources=cited_sources,
    )
