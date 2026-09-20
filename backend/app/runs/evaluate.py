"""PoC evaluation — deterministic runnability/spec checks + LLM hypothesis judge.

Runnability and spec fulfilment are decided from the sandbox result (no LLM).
Hypothesis support is one LLM call with the sandbox logs; on failure it degrades
to "not evaluated", never raises.
"""

import json
import logging

from ..core.llm import chat_json
from .schemas import EvaluationReport, HypothesisResult, PerformanceSample, WorthinessVerdict

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are an evaluation analyst. Given the hypotheses an experiment was meant to test "
    "and the sandbox logs (pytest + demo output), output JSON:\n"
    '{"hypotheses": [{"hypothesis": str, "supported": bool, "evidence": str}], '
    '"caveats": [str], "verdict": "pass|pass_with_caveats|fail", '
    '"performance_summary": str, "technical_debt": [str], "extensibility": [str], '
    '"worthiness": "worth_it|marginal|not_worth_it"}\n'
    "Rules: copy each input hypothesis verbatim into the hypothesis field, in the same "
    "order (no prefixes/rewording). supported=false when the logs do not show evidence "
    "for the hypothesis. evidence must quote the log, not paraphrase. "
    "technical_debt = shortcuts in the PoC that would need rework to ship; extensibility = "
    "what the PoC would need to grow beyond a demo; worthiness = cost/complexity vs the "
    "demonstrated benefit. Be concise. Return JSON only."
)


def _extract_demo_metrics(text: str) -> dict:
    """Pull the JSON object the demo printed (last object wins). Never raises."""
    text = (text or "").strip()
    if not text:
        return {}
    candidates: list[str] = []
    if text.startswith("{"):
        candidates.append(text)
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            candidates.append(line)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


def _worthiness_from(verdict: str, caveats: list[str], debt: list[str], llm_value: str | None) -> str:
    if llm_value in ("worth_it", "marginal", "not_worth_it"):
        return llm_value
    if verdict == "fail":
        return "not_worth_it"
    if verdict == "pass_with_caveats" or debt or caveats:
        return "marginal"
    return "worth_it"


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


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
    performance_summary = ""
    technical_debt: list[str] = []
    extensibility: list[str] = []
    llm_worthiness: str | None = None

    if hypotheses:
        user = (
            "Hypotheses:\n" + "\n".join(f"- {h}" for h in hypotheses) + "\n\n"
            f"pytest log:\n{sandbox.get('pytest_log', '')[:3000]}\n\n"
            f"demo log:\n{sandbox.get('demo_log', '')[:3000]}\n\nReturn JSON."
        )
        try:
            data = await chat_json(client, SYSTEM, user)
            returned = list(data.get("hypotheses") or [])
            by_text = {_norm(h.get("hypothesis", "")): h for h in returned}
            for index, h in enumerate(hypotheses):
                item = by_text.get(_norm(h))
                if item is None and len(returned) == len(hypotheses):
                    # prompt requires verbatim + same order; tolerate small rewordings
                    item = returned[index]
                item = item or {}
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
            performance_summary = str(data.get("performance_summary", ""))[:400]
            technical_debt = [str(x)[:300] for x in (data.get("technical_debt") or [])][:6]
            extensibility = [str(x)[:300] for x in (data.get("extensibility") or [])][:6]
            llm_worthiness = str(data.get("worthiness", "")) or None
        except Exception:
            logger.exception("hypothesis evaluation failed, degrading to not-evaluated")
            results = [HypothesisResult(hypothesis=h, supported=False, evidence="not evaluated") for h in hypotheses]
            caveats.append("hypothesis evaluation unavailable")
    else:
        results = []
        caveats.append("no hypotheses were extracted for this run")

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

    performance = PerformanceSample(
        demo_runtime_ms=int(sandbox.get("demo_runtime_ms", 0) or 0),
        sandbox_runtime_ms=int(sandbox.get("runtime_ms", 0) or 0),
        memory_peak_kb=sandbox.get("memory_peak_kb"),
        metrics=_extract_demo_metrics(str(sandbox.get("demo_log", ""))),
    )
    worthiness = WorthinessVerdict(
        worthiness=_worthiness_from(verdict, list(dict.fromkeys(caveats)), technical_debt, llm_worthiness),
        cost_usd=float(usage.get("cost_usd", 0) or 0),
        performance_summary=performance_summary,
        technical_debt=technical_debt,
        extensibility=extensibility,
    )
    return EvaluationReport(
        runnable=base["runnable"],
        sandbox_passed=base["sandbox_passed"],
        tests_passed=base["tests_passed"],
        spec_fulfilled=base["spec_fulfilled"] and all(r.supported for r in results) if results else base["spec_fulfilled"],
        hypotheses=results,
        verdict=verdict,
        caveats=list(dict.fromkeys(caveats)),
        performance=performance,
        worthiness=worthiness,
        runtime_ms=int(sandbox.get("runtime_ms", 0) or 0),
        wall_ms=wall_ms,
        llm_calls=int(usage.get("calls", 0) or 0),
        cost_usd=float(usage.get("cost_usd", 0) or 0),
    )
