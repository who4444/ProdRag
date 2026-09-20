"""Integration plan — turn a passing PoC into an apply proposal (changes.md).

Proposal only: the target repo is never modified. One LLM call, deterministic
minimal fallback, markdown renderer so the plan is human-readable.
"""

import logging

from ..core.llm import chat_json
from .schemas import FileChange, IntegrationPlan

logger = logging.getLogger(__name__)

MAX_STEPS = 8
MAX_RISKS = 6
MAX_VERIFICATION = 6
MAX_FILES = 8

SYSTEM = (
    "You are an integration planner. Given an idea, a cited feasibility report, an "
    "experiment plan, and the files of a passing isolated PoC, propose how to apply the "
    "change to the target codebase. Output JSON:\n"
    '{"summary": str, "files": [{"path": str, "change_type": "add|modify", '
    '"rationale": str, "sketch": str}], "steps": [str], "risks": [str], '
    '"verification": [str]}\n'
    "Rules:\n"
    "- files[].path must be a real target path from the extension points / impacted "
    "modules, not the PoC's demo file names.\n"
    "- sketch is a short pseudocode or diff-style fragment, not a full file.\n"
    "- verification lists concrete commands/tests to run in the target repo.\n"
    "- This is a proposal: do not claim the change is applied.\n"
    "Be concise. Return JSON only."
)


def _fallback_plan(idea: str, report: dict, poc_plan: dict, file_paths: list[str]) -> IntegrationPlan:
    extension_points = (report.get("codebase_map") or {}).get("extension_points") or []
    files = [
        FileChange(
            path=str(ep.get("path", "")),
            change_type="modify",
            rationale=str(ep.get("reason", ""))[:300] or "extension point from the feasibility report",
            sketch=f"# Port logic from the PoC file(s): {', '.join(file_paths[:4]) or 'n/a'}",
        )
        for ep in extension_points[:MAX_FILES]
        if ep.get("path")
    ]
    steps = [
        "Map the PoC's cache/logic into the extension point(s) above.",
        "Keep the PoC's JSON metrics contract for testing.",
        "Run the verification commands below in the target repo.",
    ]
    if poc_plan.get("system_context"):
        steps.append(f"Context: {poc_plan['system_context'][:200]}")
    return IntegrationPlan(
        summary=f"Apply the PoC for '{idea[:150]}' at the cited extension points (manual review required).",
        files=files,
        steps=steps,
        risks=["Automatic plan unavailable — manual review of the PoC is required before applying."],
        verification=["Run the repository test suite.", "Run the PoC's demo and compare its JSON metrics."],
    )


def _sanitize(data: dict, fallback: IntegrationPlan) -> IntegrationPlan:
    files: list[FileChange] = []
    for item in (data.get("files") or [])[:MAX_FILES]:
        if not isinstance(item, dict) or not str(item.get("path", "")).strip():
            continue
        change_type = item.get("change_type") if item.get("change_type") in ("add", "modify") else "modify"
        files.append(
            FileChange(
                path=str(item["path"])[:300],
                change_type=change_type,
                rationale=str(item.get("rationale", ""))[:400],
                sketch=str(item.get("sketch", ""))[:1200],
            )
        )
    return IntegrationPlan(
        summary=str(data.get("summary") or fallback.summary)[:800],
        files=files or fallback.files,
        steps=[str(x)[:300] for x in (data.get("steps") or [])][:MAX_STEPS] or fallback.steps,
        risks=[str(x)[:300] for x in (data.get("risks") or [])][:MAX_RISKS] or fallback.risks,
        verification=[str(x)[:300] for x in (data.get("verification") or [])][:MAX_VERIFICATION]
        or fallback.verification,
    )


def render_markdown(plan: IntegrationPlan, idea: str) -> str:
    lines = [f"# Integration plan: {idea[:160]}", "", plan.summary or "(no summary)", "", "## Files"]
    if plan.files:
        for f in plan.files:
            lines.append(f"### `{f.path}` ({f.change_type})")
            if f.rationale:
                lines.append(f.rationale)
            if f.sketch:
                lines += ["", "```python", f.sketch, "```"]
            lines.append("")
    else:
        lines.append("(none proposed — manual mapping required)")
    for title, items in (("Steps", plan.steps), ("Risks", plan.risks), ("Verification", plan.verification)):
        lines += [f"## {title}"]
        lines += [f"- {item}" for item in items] or ["- (none)"]
        lines.append("")
    lines.append("_Proposal only — the target repository was not modified._")
    return "\n".join(lines)


async def build_integration_plan(
    client,
    idea: str,
    report: dict,
    poc_plan: dict,
    file_contents: list[dict],
    sandbox: dict,
) -> IntegrationPlan:
    """Build the apply proposal. Never raises."""
    file_paths = [str(f.get("path", "")) for f in file_contents if f.get("path")]
    fallback = _fallback_plan(idea, report, poc_plan or {}, file_paths)
    if not sandbox.get("passed"):
        return fallback

    files_block = "\n\n".join(
        f"--- {f.get('path')} ---\n{(f.get('content') or '')[:1200]}" for f in file_contents[:6]
    )
    extension_points = ", ".join(
        f"{ep.get('path')}::{ep.get('symbol', '')}"
        for ep in (report.get("codebase_map") or {}).get("extension_points") or []
    ) or "none"
    claims = "; ".join(
        c.get("statement", "") for c in (report.get("claims") or []) if c.get("status") == "supported"
    )[:600] or "none"
    user = (
        f"Idea: {idea[:500]}\n"
        f"Experiment plan: {str(poc_plan.get('chosen_method', ''))[:400]}\n"
        f"Supported claims: {claims}\n"
        f"Extension points: {extension_points}\n"
        f"PoC sandbox: passed={sandbox.get('passed')} tests={str(sandbox.get('pytest_log', ''))[:120]}\n\n"
        f"PoC files:\n{files_block}\n\nReturn JSON."
    )
    try:
        data = await chat_json(client, SYSTEM, user)
        plan = _sanitize(data, fallback)
    except Exception:
        logger.exception("integration plan LLM failed, using deterministic fallback")
        plan = fallback
    plan.changes_md = render_markdown(plan, idea)
    return plan
