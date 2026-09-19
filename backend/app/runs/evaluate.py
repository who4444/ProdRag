"""PoC evaluation — deterministic runnability/spec checks + LLM hypothesis judge.

Runnability and spec fulfilment are decided from the sandbox result (no LLM).
Hypothesis support is one LLM call with the sandbox logs; on failure it degrades
to "not evaluated", never raises.
"""

import logging

from ..core.llm import chat_json
from .schemas import EvaluationReport, HypothesisResult

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are an evaluation analyst. Given the hypotheses an experiment was meant to test "
    "and the sandbox logs (pytest + demo output), output JSON:\n"
    '{"hypotheses": [{"hypothesis": str, "supported": bool, "evidence": str}], '
    '"caveats": [str], "verdict": "pass|pass_with_caveats|fail"}\n'
    "Rules: supported=false when the logs do not show evidence for the hypothesis. "
    "evidence must quote the log, not paraphrase. Be concise. Return JSON only."
)


def _hypotheses_from(understanding: dict | None) -> list[str]:
    return [str(h) for h in ((understanding or {}).get("hypotheses") or [])][:6]


def _deterministic_report(sandbox: dict) -> dict:
    passed = bool((sandbox or {}).get("passed"))
    pytest_log = str((sandbox or {}).get("pytest_log", ""))
    tests_passed = passed and ("passed" in pytest_log.lower() or "no test file" in pytest_log.lower())
    caveats = []
    if not passed:
        caveats.append(f"sandbox failed (returncode={sandbox.get('returncode')})")
    if passed and not tests_passed:
        caveats.append("pytest did not pass cleanly")
    return {
        "runnable": passed,
        "sandbox_passed": passed,
        "tests_passed": tests_passed,
        "spec_fulfilled": passed and tests_passed,
        "hypotheses": [],
        "verdict": "pass" if (passed and tests_passed) else "fail",
        "caveats": caveats,
    }


async def evaluate_poc(
    client,
    understanding: dict | None,
    sandbox: dict,
    usage: dict,
    wall_ms: int,
) -> EvaluationReport:
    """Build the EvaluationReport. Never raises."""
    base = _deterministic_report(sandbox)
    hypotheses = _hypotheses_from(understanding)
    results: list[HypothesisResult] = []
    llm_verdict: str | None = None
    caveats = list(base["caveats"])

    if hypotheses:
        user = (
            "Hypotheses:\n" + "\n".join(f"- {h}" for h in hypotheses) + "\n\n"
            f"pytest log:\n{sandbox.get('pytest_log', '')[:3000]}\n\n"
            f"demo log:\n{sandbox.get('demo_log', '')[:3000]}\n\nReturn JSON."
        )
        try:
            data = await chat_json(client, SYSTEM, user)
            by_text = {str(h.get("hypothesis", "")).strip(): h for h in (data.get("hypotheses") or [])}
            for h in hypotheses:
                item = by_text.get(h) or {}
                results.append(
                    HypothesisResult(
                        hypothesis=h,
                        supported=bool(item.get("supported", False)),
                        evidence=str(item.get("evidence", ""))[:300] or "not evaluated",
                    )
                )
            caveats.extend(str(c)[:300] for c in (data.get("caveats") or [])[:5])
            v = str(data.get("verdict", ""))
            if v in ("pass", "pass_with_caveats", "fail"):
                llm_verdict = v
        except Exception:
            logger.exception("hypothesis evaluation failed, degrading to not-evaluated")
            results = [HypothesisResult(hypothesis=h, supported=False, evidence="not evaluated") for h in hypotheses]
            caveats.append("hypothesis evaluation unavailable")
    else:
        results = []

    if not base["sandbox_passed"]:
        verdict = "fail"
    elif llm_verdict:
        verdict = llm_verdict
    elif hypotheses and not all(r.supported for r in results):
        verdict = "pass_with_caveats"
    elif not base["tests_passed"]:
        verdict = "pass_with_caveats"
    else:
        verdict = "pass"

    return EvaluationReport(
        runnable=base["runnable"],
        sandbox_passed=base["sandbox_passed"],
        tests_passed=base["tests_passed"],
        spec_fulfilled=base["spec_fulfilled"] and all(r.supported for r in results) if results else base["spec_fulfilled"],
        hypotheses=results,
        verdict=verdict,
        caveats=list(dict.fromkeys(caveats)),
        runtime_ms=int(sandbox.get("runtime_ms", 0) or 0),
        wall_ms=wall_ms,
        llm_calls=int(usage.get("calls", 0) or 0),
        cost_usd=float(usage.get("cost_usd", 0) or 0),
    )
