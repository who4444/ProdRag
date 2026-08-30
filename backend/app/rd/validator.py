"""Validator — LLM-as-judge, emit-with-risks (never blocks)."""

import json
import logging

from ..config import settings
from .schemas import Requirements, ResearchSummary, Validation

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a research validator. Given structured requirements and a list of "
    "research summaries (one per direction), check coverage and feasibility for "
    "a runnable demo. Output JSON only with shape:\n"
    '{"coverage": {"core_algorithm": bool, ...}, "missing": ["..."], "feasible": bool, "risks": ["..."]}\n'
    "Rules: coverage true if the direction has substantive findings (not just gaps). "
    "missing lists uncovered requirements. feasible false only if a blocker prevents a toy demo. "
    "risks lists caveats even when feasible. Be concise."
)


def _fallback_validation(requirements: Requirements, summaries: list[ResearchSummary]) -> Validation:
    coverage = {s.direction_id: s.confidence != "low" for s in summaries}
    missing = [s.direction_id for s in summaries if s.confidence == "low"]
    return Validation(coverage=coverage, missing=missing, feasible=True, risks=missing and [f"Low evidence for: {', '.join(missing)}"] or [])


async def validate(client, requirements: Requirements, summaries: list[ResearchSummary]) -> Validation:
    payload = {
        "requirements": requirements.model_dump(),
        "summaries": [
            {"direction_id": s.direction_id, "findings": s.findings[:1200], "confidence": s.confidence, "gaps": s.gaps}
            for s in summaries
        ],
    }
    try:
        resp = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(payload)},
            ],
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        return Validation(
            coverage={str(k): bool(v) for k, v in (data.get("coverage") or {}).items()},
            missing=[str(x) for x in (data.get("missing") or [])][:10],
            feasible=bool(data.get("feasible", True)),
            risks=[str(x) for x in (data.get("risks") or [])][:10],
        )
    except Exception:
        logger.exception("validator failed, falling back")
        return _fallback_validation(requirements, summaries)
