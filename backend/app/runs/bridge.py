"""Feasibility report -> ResearchArtifact bridge.

The code pipeline speaks ResearchArtifact; the feasibility phase speaks
FeasibilityReport. This translator carries the cited claims, extension points,
and hypotheses across so the PoC is conditioned on feasibility evidence.
"""

import time

from ..rd.schemas import DemoSpec, Requirements, ResearchArtifact, ResearchSummary, Validation

DIMENSIONS = ("technical", "integration", "data")


def report_to_artifact(report: dict, idea: str) -> ResearchArtifact:
    data = report or {}
    understanding = data.get("understanding") or {}
    claims = data.get("claims") or []
    dimensions = data.get("dimensions") or {}
    codebase_map = data.get("codebase_map") or {}

    summaries: list[ResearchSummary] = []
    for dim in DIMENSIONS:
        dim_claims = [c for c in claims if c.get("dimension") == dim]
        lines = []
        for claim in dim_claims:
            refs = ", ".join(
                f"{r.get('path')}:{r.get('start_line')}-{r.get('end_line')}"
                for r in claim.get("evidence", [])
                if r.get("valid")
            )
            lines.append(f"[{claim.get('status')}] {claim.get('statement')}" + (f" ({refs})" if refs else ""))
        supported = any(c.get("status") == "supported" for c in dim_claims)
        summaries.append(
            ResearchSummary(
                direction_id=dim,
                findings="\n".join(lines)[:2000] or "No evidence-backed findings.",
                sources=[],
                confidence="high" if supported else "low",
                gaps=[] if supported else [f"no supported evidence for {dim}"],
            )
        )

    coverage = {dim: bool(dimensions.get(dim, {}).get("plausible")) for dim in DIMENSIONS}
    validation = Validation(
        coverage=coverage,
        missing=[dim for dim, ok in coverage.items() if not ok],
        feasible=data.get("verdict") != "infeasible",
        risks=list(data.get("blockers") or []),
    )

    supported_technical = [
        c.get("statement", "") for c in claims if c.get("dimension") == "technical" and c.get("status") == "supported"
    ]
    core_algorithm = "\n".join(supported_technical) or understanding.get("interpretation", "")
    extension_points = codebase_map.get("extension_points") or []
    pseudocode = "\n".join(
        f"- {ep.get('path')}::{ep.get('symbol', '')} {ep.get('reason', '')}" for ep in extension_points[:5]
    )
    hypotheses = understanding.get("hypotheses") or []
    criteria = [f"PoC hypothesis: {h}" for h in hypotheses[:4]] + ["demo.py exits 0"]

    demo_spec = DemoSpec(
        entrypoint="demo.py",
        tech_stack=["python"],
        core_algorithm=core_algorithm[:1500],
        pseudocode=pseudocode[:1500],
        acceptance_criteria=criteria,
    )
    requirements = Requirements(
        goal=idea[:300],
        core_idea=(understanding.get("interpretation") or idea)[:300],
        demo_type="script",
        constraints=[],
        audience="engineer",
        open_questions=[],
    )
    return ResearchArtifact(
        requirements=requirements,
        summaries=summaries,
        validation=validation,
        demo_spec=demo_spec,
        sources=[],
        created_at=time.time(),
    )
