"""Judge — dimension verdicts + evidence gate + final verdict.

The LLM writes rationale/blockers/alternatives/reframe; the deterministic gate
decides which dimensions may count as plausible and maps that to the verdict:

  0 dimensions not plausible  -> plausible (or plausible_with_risks if risks)
  1 dimension not plausible   -> plausible_with_risks (reframe path)
  2+ dimensions not plausible -> infeasible

Invariant: a dimension cannot be plausible without at least one claim with a
valid citation. Emit-with-risks: the judge never blocks a report.
"""

import logging

from ..core.llm import chat_json
from .schemas import (
    DIMENSION_IDS,
    CodebaseMap,
    DimensionVerdict,
    FeasibilityClaim,
    IdeaUnderstanding,
)

logger = logging.getLogger(__name__)

FALLBACK_ALTERNATIVES = [
    "Narrow the scope to the modules with evidence, then re-run feasibility.",
    "Prototype against the closest existing extension point instead of new architecture.",
]

SYSTEM = (
    "You are a feasibility judge. Given an idea, a codebase map, and evidence-backed claims "
    "per dimension (technical, integration, data), output JSON:\n"
    '{"dimensions": {"technical": {"plausible": bool, "rationale": str, "blockers": [str], '
    '"missing_evidence": [str]}, "integration": {...}, "data": {...}}, '
    '"verdict": "plausible|plausible_with_risks|infeasible", "blockers": [str], '
    '"alternatives": [str], "reframed_idea": str|null}\n'
    "Rules:\n"
    "- A dimension is plausible only if its claims are supported by code evidence.\n"
    "- blockers are concrete reasons an implementation would fail or need rework.\n"
    "- alternatives are evidence-based suggestions, not generic advice; reframed_idea "
    "rewrites the idea to fit the codebase when the original is not feasible.\n"
    "Be concise. Return JSON only."
)


def _claims_view(claims: list[FeasibilityClaim]) -> str:
    lines = []
    for c in claims[:20]:
        refs = ", ".join(
            f"{r.path}:{r.start_line}-{r.end_line}" if r.start_line else r.path for r in c.evidence if r.valid
        )
        lines.append(f"- [{c.dimension}/{c.status}] {c.statement[:220]}" + (f" | evidence: {refs}" if refs else ""))
    return "\n".join(lines) or "(no claims)"


def apply_evidence_gate(
    claims: list[FeasibilityClaim], dimensions: dict[str, DimensionVerdict], llm_verdict: str
) -> tuple[dict[str, DimensionVerdict], str, list[str]]:
    """Enforce evidence, force a not-plausible dimension when unsupported, map verdict."""
    blockers: list[str] = []
    for dim in DIMENSION_IDS:
        verdict = dimensions.setdefault(dim, DimensionVerdict())
        supported = [c for c in claims if c.dimension == dim and c.status == "supported"]
        if not supported:
            verdict.plausible = False
            if not verdict.missing_evidence:
                verdict.missing_evidence = [f"no valid code evidence for {dim}"]
        if not verdict.plausible:
            if not verdict.blockers:
                reason = verdict.missing_evidence[0] if verdict.missing_evidence else "not plausible"
                verdict.blockers = [f"{dim}: {reason}"]
            blockers.extend(verdict.blockers)

    implausible = [d for d in DIMENSION_IDS if not dimensions[d].plausible]
    if len(implausible) >= 2:
        verdict = "infeasible"
    elif len(implausible) == 1:
        verdict = "plausible_with_risks"
    else:
        risks_present = llm_verdict == "plausible_with_risks" or bool(blockers)
        verdict = "plausible_with_risks" if risks_present else "plausible"
    return dimensions, verdict, list(dict.fromkeys(blockers))


def _fallback_verdicts(claims: list[FeasibilityClaim]) -> dict[str, DimensionVerdict]:
    dims: dict[str, DimensionVerdict] = {}
    for dim in DIMENSION_IDS:
        supported = [c for c in claims if c.dimension == dim and c.status == "supported"]
        if supported:
            dims[dim] = DimensionVerdict(
                plausible=True,
                rationale=f"{len(supported)} claim(s) supported by code evidence",
                missing_evidence=[],
            )
        else:
            dims[dim] = DimensionVerdict(
                plausible=False,
                rationale="No supported code evidence",
                missing_evidence=[f"no valid code evidence for {dim}"],
            )
    return dims


async def judge_feasibility(
    client,
    idea: str,
    understanding: IdeaUnderstanding,
    cmap: CodebaseMap,
    claims: list[FeasibilityClaim],
) -> dict:
    """Returns {dimensions, verdict, blockers, alternatives, reframed_idea}. Never raises."""
    user = (
        f"Idea: {idea[:500]}\n"
        f"Interpretation: {understanding.interpretation[:400]}\n"
        f"Hypotheses: {'; '.join(understanding.hypotheses[:4]) or 'none'}\n"
        f"Impacted modules: {', '.join(cmap.impacted_modules[:15]) or 'none'}\n"
        f"Extension points: {', '.join(ep.path for ep in cmap.extension_points[:5]) or 'none'}\n"
        f"Data available: {', '.join(cmap.data_available[:10]) or 'none'} | missing: {', '.join(cmap.data_missing[:10]) or 'none'}\n"
        f"Data cleanliness: {cmap.data_cleanliness}\n\n"
        f"Claims:\n{_claims_view(claims)}\n\nReturn JSON."
    )

    dims: dict[str, DimensionVerdict]
    llm_verdict = "infeasible"
    alternatives: list[str] = []
    reframed: str | None = None
    try:
        data = await chat_json(client, SYSTEM, user)
        raw_dims = data.get("dimensions") or {}
        dims = {}
        for dim in DIMENSION_IDS:
            entry = raw_dims.get(dim) or {}
            dims[dim] = DimensionVerdict(
                plausible=bool(entry.get("plausible", False)),
                rationale=str(entry.get("rationale", ""))[:500],
                blockers=[str(x)[:300] for x in (entry.get("blockers") or [])][:5],
                missing_evidence=[str(x)[:300] for x in (entry.get("missing_evidence") or [])][:5],
            )
        llm_verdict = str(data.get("verdict", "infeasible"))
        alternatives = [str(x)[:300] for x in (data.get("alternatives") or [])][:5]
        reframed = data.get("reframed_idea")
        reframed = str(reframed)[:500] if reframed else None
    except Exception:
        logger.exception("judge LLM failed, using deterministic verdicts")
        dims = _fallback_verdicts(claims)

    dims, verdict, blockers = apply_evidence_gate(claims, dims, llm_verdict)
    if verdict != "plausible" and not alternatives:
        alternatives = list(FALLBACK_ALTERNATIVES)
    return {
        "dimensions": {k: v.model_dump() for k, v in dims.items()},
        "verdict": verdict,
        "blockers": blockers,
        "alternatives": alternatives,
        "reframed_idea": reframed,
    }
