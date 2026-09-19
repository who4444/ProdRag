"""Dimension subagent — code_search -> cited claims.

One async function x3 (technical | integration | data), fan-out via asyncio.gather.
Never raises: on LLM failure it returns a low-confidence claim carrying the raw
retrieval so the judge can still reason about missing evidence.
"""

import logging
from pathlib import Path

from ..core.llm import chat_json
from ..tools.code_rag import code_search
from .citation import apply_citation_gate
from .schemas import CodebaseMap, EvidenceRef, FeasibilityClaim, IdeaUnderstanding

logger = logging.getLogger(__name__)

MAX_CLAIMS = 4
MAX_EVIDENCE_PER_CLAIM = 4
HIT_TEXT = 900

SYSTEM = (
    "You are a feasibility analyst for an existing codebase. Given an idea and retrieved "
    "code excerpts, output JSON:\n"
    '{"claims": [{"statement": str, "confidence": "low|medium|high", '
    '"evidence": [{"path": str, "start_line": int, "end_line": int}]}]}\n'
    "Rules:\n"
    "- Every claim must cite at least one excerpt with its exact path and line numbers.\n"
    "- If the excerpts do not support a claim, state that in a claim with no evidence.\n"
    "- 2-4 claims, specific to the dimension. Never invent paths or line numbers.\n"
    "Return JSON only."
)


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_hits(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(
            f"[{i}] {h.get('path')}::{h.get('symbol', '')} "
            f"(lines {h.get('start_line')}-{h.get('end_line')}, kind={h.get('kind')})\n"
            f"{h.get('text', '')[:HIT_TEXT]}"
        )
    return "\n\n".join(parts) or "No code excerpts retrieved."


async def assess_dimension(
    client,
    dimension: dict,
    idea: str,
    understanding: IdeaUnderstanding,
    cmap: CodebaseMap,
    root: Path,
    repo_id: str,
    k: int = 6,
) -> tuple[list[FeasibilityClaim], list[dict]]:
    """Returns (gated claims, raw hits) for one dimension."""
    question = dimension["question"]
    try:
        result = await code_search(question, k=k, repo_id=repo_id)
    except Exception:
        logger.exception("code_search failed for dimension %s", dimension["id"])
        result = {"hits": []}
    hits = result.get("hits", [])

    user = (
        f"Dimension: {dimension['id']}\n"
        f"Question: {question}\n"
        f"Idea: {idea[:400]}\n"
        f"Interpretation: {understanding.interpretation[:300]}\n"
        f"Impacted modules: {', '.join(cmap.impacted_modules[:12]) or 'none'}\n\n"
        f"Retrieved code:\n{_format_hits(hits)}\n\nReturn JSON."
    )

    claims: list[FeasibilityClaim] = []
    try:
        data = await chat_json(client, SYSTEM, user)
        for i, item in enumerate((data.get("claims") or [])[:MAX_CLAIMS]):
            evidence = [
                EvidenceRef(
                    path=str(e.get("path", "")),
                    start_line=_as_int(e.get("start_line")),
                    end_line=_as_int(e.get("end_line")),
                )
                for e in (item.get("evidence") or [])[:MAX_EVIDENCE_PER_CLAIM]
            ]
            statement = str(item.get("statement", ""))[:500]
            if not statement:
                continue
            claims.append(
                FeasibilityClaim(
                    id=f"{dimension['id']}-{i + 1}",
                    dimension=dimension["id"],
                    statement=statement,
                    evidence=evidence,
                    confidence=item.get("confidence") if item.get("confidence") in ("low", "medium", "high") else "low",
                )
            )
    except Exception:
        logger.exception("dimension %s assessment failed", dimension["id"])

    if not claims:
        fallback_evidence = [
            EvidenceRef(path=h["path"], start_line=h.get("start_line"), end_line=h.get("end_line"))
            for h in hits[:2]
        ]
        claims = [
            FeasibilityClaim(
                id=f"{dimension['id']}-fallback",
                dimension=dimension["id"],
                statement=(
                    "Assessment unavailable (LLM failure); raw retrieval collected for manual review."
                    if not hits
                    else "Assessment unavailable (LLM failure); candidate code retrieved for manual review."
                ),
                evidence=fallback_evidence,
                confidence="low",
                provisional=True,
            )
        ]

    return apply_citation_gate(root, claims), hits
