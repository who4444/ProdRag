"""PoC plan — one LLM call turning feasibility evidence into an experiment design.

Lean research agent for the run graph: chosen method, implementation requirements,
experiment objective, evaluation metrics, system context. The knowledge base is
consulted opportunistically (one rag_search) and skipped when empty/unavailable;
the heavyweight rd/ 5-direction pipeline stays separate for paper -> demo runs.
Never raises: a deterministic fallback plan is built from the report.
"""

import logging

from ..core.llm import chat_json
from .schemas import EvaluationMetric, PocPlan

logger = logging.getLogger(__name__)

MAX_CLAIMS = 8
MAX_KB_HITS = 3
MAX_REQUIREMENTS = 6
MAX_METRICS = 6

SYSTEM = (
    "You are an R&D experiment planner. Given an idea, a cited feasibility report, and a "
    "codebase map, output JSON:\n"
    '{"chosen_method": str, "implementation_requirements": [str], '
    '"experiment_objective": str, '
    '"evaluation_metrics": [{"name": str, "direction": "increase|decrease|present|bounded", '
    '"threshold": str}], "system_context": str}\n'
    "Rules:\n"
    "- chosen_method: the concrete approach to prototype, grounded in the cited evidence and "
    "extension points (no new dependencies unless evidence requires them).\n"
    "- implementation_requirements: what the isolated PoC must contain/measure.\n"
    "- evaluation_metrics: machine-checkable metrics the demo should print (2-5), each with a "
    "direction and a threshold. Metric names must be simple lowercase snake_case JSON keys "
    "(e.g. cache_hits, cache_misses, embedding_calls, search_kb_calls) — no long "
    "descriptive names.\n"
    "- system_context: where this would live in the target codebase and what it touches.\n"
    "Be concise. Return JSON only."
)


def _supported_claims(report: dict) -> list[dict]:
    claims = [c for c in (report.get("claims") or []) if c.get("status") == "supported"]
    return claims[:MAX_CLAIMS]


def _claims_view(report: dict) -> str:
    lines = []
    for c in _supported_claims(report):
        refs = ", ".join(
            f"{r.get('path')}:{r.get('start_line')}-{r.get('end_line')}"
            for r in c.get("evidence", [])
            if r.get("valid")
        )
        lines.append(f"- [{c.get('dimension')}] {c.get('statement')}" + (f" ({refs})" if refs else ""))
    return "\n".join(lines) or "(no supported claims)"


def _kb_context(hits: list[dict]) -> str:
    if not hits:
        return ""
    parts = []
    for i, h in enumerate(hits[:MAX_KB_HITS], 1):
        parts.append(f"[{i}] {h.get('source', '')} p{h.get('page', '')}: {str(h.get('text', ''))[:300]}")
    return "\n\nKnowledge base hits:\n" + "\n".join(parts)


async def _kb_hits(idea: str, k: int) -> list[dict]:
    try:
        from ..tools.rag import rag_search

        result = await rag_search(idea, k=min(k, MAX_KB_HITS), k_images=0)
        return result.get("text") or []
    except Exception:
        logger.debug("poc_plan: KB lookup skipped", exc_info=True)
        return []


def _fallback_plan(idea: str, understanding: dict, report: dict, cmap: dict) -> PocPlan:
    interpretation = (understanding or {}).get("interpretation") or idea
    hypotheses = (understanding or {}).get("hypotheses") or []
    extension_points = (report.get("codebase_map") or cmap or {}).get("extension_points") or []
    impacted = (cmap or {}).get("impacted_modules") or []
    requirements = [
        f"Target {ep.get('path')}::{ep.get('symbol', '')} — {ep.get('reason', 'extension point')}"
        for ep in extension_points[:MAX_REQUIREMENTS]
    ]
    metrics = [
        EvaluationMetric(name=f"hypothesis_{i + 1}", direction="present", threshold=h[:120])
        for i, h in enumerate(hypotheses[:MAX_METRICS])
    ]
    return PocPlan(
        chosen_method=interpretation[:600],
        implementation_requirements=requirements,
        experiment_objective=(hypotheses[0] if hypotheses else interpretation)[:400],
        evaluation_metrics=metrics,
        system_context=", ".join(impacted[:8]) or "unknown",
    )


def _sanitize(data: dict, fallback: PocPlan) -> PocPlan:
    metrics: list[EvaluationMetric] = []
    for item in (data.get("evaluation_metrics") or [])[:MAX_METRICS]:
        if not isinstance(item, dict) or not str(item.get("name", "")).strip():
            continue
        direction = item.get("direction")
        if direction not in ("increase", "decrease", "present", "bounded"):
            direction = "present"
        metrics.append(
            EvaluationMetric(
                name=str(item["name"])[:120],
                direction=direction,
                threshold=str(item.get("threshold", ""))[:200],
            )
        )
    requirements = [str(x)[:300] for x in (data.get("implementation_requirements") or []) if str(x).strip()]
    return PocPlan(
        chosen_method=str(data.get("chosen_method") or fallback.chosen_method)[:800],
        implementation_requirements=requirements[:MAX_REQUIREMENTS] or fallback.implementation_requirements,
        experiment_objective=str(data.get("experiment_objective") or fallback.experiment_objective)[:500],
        evaluation_metrics=metrics or fallback.evaluation_metrics,
        system_context=str(data.get("system_context") or fallback.system_context)[:600],
    )


async def build_poc_plan(
    client,
    idea: str,
    understanding: dict,
    report: dict,
    cmap: dict,
    k: int = 3,
) -> PocPlan:
    """Build the experiment plan. Never raises."""
    fallback = _fallback_plan(idea, understanding, report, cmap)
    hits = await _kb_hits(idea, k)
    extension_points = ", ".join(
        f"{ep.get('path')}::{ep.get('symbol', '')}" for ep in (cmap or {}).get("extension_points") or []
    ) or "none"
    impacted = ", ".join((cmap or {}).get("impacted_modules") or [])[:12] or "none"
    data_available = ", ".join((cmap or {}).get("data_available") or [])[:8] or "none"
    data_missing = ", ".join((cmap or {}).get("data_missing") or [])[:8] or "none"
    hypotheses = "; ".join((understanding or {}).get("hypotheses") or [])[:600] or "none"
    user = (
        f"Idea: {idea[:600]}\n"
        f"Interpretation: {(understanding or {}).get('interpretation', '')[:400]}\n"
        f"Hypotheses: {hypotheses}\n\n"
        f"Verified feasibility claims:\n{_claims_view(report)}\n\n"
        f"Extension points: {extension_points}\n"
        f"Impacted modules: {impacted}\n"
        f"Blockers: {'; '.join(report.get('blockers') or [])[:400] or 'none'}\n"
        f"Data: available={data_available}; missing={data_missing}"
        f"{_kb_context(hits)}\n\nReturn JSON."
    )
    try:
        data = await chat_json(client, SYSTEM, user)
        return _sanitize(data, fallback)
    except Exception:
        logger.exception("poc_plan LLM failed, using deterministic fallback")
        return fallback
