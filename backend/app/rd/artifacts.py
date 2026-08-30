"""Artifact builder — handoff to the coding module."""

import time

from .schemas import DemoSpec, ResearchArtifact, Requirements, ResearchSummary, Validation


def dedupe_sources(summaries: list[ResearchSummary]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for s in summaries:
        for item in s.sources:
            key = f"{item.get('source')}|{item.get('page')}|{item.get('content','')[:80]}"
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


def build_demo_spec(requirements: Requirements, summaries: list[ResearchSummary], validation: Validation) -> DemoSpec:
    # Collect core algorithm excerpt from the relevant summary
    core = next((s.findings for s in summaries if s.direction_id == "core_algorithm"), "")
    eval_findings = next((s.findings for s in summaries if s.direction_id == "evaluation"), "")

    pseudocode = core[:800] if core else requirements.core_idea

    criteria = [
        "demo.py exits with code 0 on toy data",
        "core idea is demonstrated (output is non-trivial and matches expected shape)",
    ]
    if eval_findings:
        criteria.append(f"toy check passes: {eval_findings[:120]}")
    if validation.risks:
        criteria.append(f"known risks acknowledged: {validation.risks[0][:100]}")

    tech_stack = ["python", "numpy"]
    if any("torch" in s.findings.lower() or "pytorch" in s.findings.lower() for s in summaries):
        tech_stack.append("torch")
    if "pandas" in " ".join(s.findings.lower() for s in summaries):
        if "pandas" not in tech_stack:
            tech_stack.append("pandas")

    return DemoSpec(
        entrypoint="demo.py",
        tech_stack=tech_stack,
        core_algorithm=core[:1200] or requirements.core_idea,
        pseudocode=pseudocode,
        acceptance_criteria=criteria[:5],
    )


def build_artifact(
    requirements: Requirements,
    summaries: list[ResearchSummary],
    validation: Validation,
) -> ResearchArtifact:
    return ResearchArtifact(
        requirements=requirements,
        summaries=summaries,
        validation=validation,
        demo_spec=build_demo_spec(requirements, summaries, validation),
        sources=dedupe_sources(summaries),
        created_at=time.time(),
    )
