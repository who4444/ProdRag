"""Graph nodes — thin wrappers over feasibility/, code/, and evaluation helpers.

Every node activates the run's usage ledger, emits NDJSON events through the
LangGraph stream writer, and appends the same payload to the evidence ledger.
Nodes never raise for expected failures: they degrade to emit-with-risks.
"""

import asyncio
import logging
import time
from pathlib import Path

from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from ..code.pipeline import code_generate
from ..core import db, llm
from ..feasibility.citation import apply_citation_gate
from ..feasibility.codebase_map import enrich_map, static_map
from ..feasibility.judge import judge_feasibility
from ..feasibility.planner import build_dimensions
from ..feasibility.schemas import (
    EvidenceRef,
    FeasibilityClaim,
    FeasibilityReport,
    IdeaUnderstanding,
)
from ..feasibility.subagent import assess_dimension
from ..feasibility.understand import understand_idea
from .bridge import report_to_artifact
from .evaluate import evaluate_poc
from .schemas import RunResult
from .state import RunState

logger = logging.getLogger(__name__)

DIMENSIONS = ("technical", "integration", "data")


def _client(config):
    return config["configurable"]["client"]


def _write(payload: dict) -> None:
    try:
        get_stream_writer()(payload)
    except Exception:  # outside graph execution / tests
        logger.debug("stream writer unavailable for %s", payload.get("event"))


def _emit(payload: dict, **updates) -> dict:
    _write(payload)
    return {"events": [payload], **updates}


def budget_ok(state: RunState) -> bool:
    usage = state.get("usage") or {}
    calls = int(usage.get("calls", 0) or 0)
    if calls >= int(state.get("max_llm_calls", 25)):
        return False
    elapsed = time.time() - float(state.get("started_at", time.time()))
    return elapsed < int(state.get("max_wall_s", 900))


def _understanding(state: RunState) -> IdeaUnderstanding:
    return IdeaUnderstanding(**(state.get("understanding") or {}))


async def understand_node(state: RunState, config) -> dict:
    run_id = state["run_id"]
    usage = llm.activate_usage(run_id)
    if state.get("understanding"):
        return _emit(
            {"type": "run", "event": "state", "phase": "understand", "status": "cached", "run_id": run_id},
            usage=usage.snapshot(),
        )
    result = await understand_idea(_client(config), state.get("idea", ""), state.get("answers"))
    if not result["ready"]:
        return _emit(
            {
                "type": "feasibility",
                "event": "clarify",
                "run_id": run_id,
                "questions": result["questions"] or [],
            },
            questions=result["questions"] or [],
            status="clarifying",
            usage=usage.snapshot(),
        )
    understanding = result["understanding"].model_dump()
    return _emit(
        {"type": "feasibility", "event": "understand", "run_id": run_id, "understanding": understanding},
        understanding=understanding,
        questions=[],
        answers=[],
        status="running",
        usage=usage.snapshot(),
    )


async def clarify_node(state: RunState, config) -> dict:
    run_id = state["run_id"]
    payload = {
        "type": "run",
        "event": "interrupt",
        "kind": "clarify",
        "run_id": run_id,
        "questions": state.get("questions", []),
    }
    resumed = interrupt(payload)
    if isinstance(resumed, dict):
        answers = resumed.get("answers", [])
    elif isinstance(resumed, list):
        answers = resumed
    else:
        answers = [str(resumed)]
    return _emit(
        {"type": "run", "event": "state", "phase": "clarify", "status": "resumed", "run_id": run_id},
        answers=[str(a) for a in answers],
    )


async def codebase_map_node(state: RunState, config) -> dict:
    run_id = state["run_id"]
    usage = llm.activate_usage(run_id)
    root = Path(state["repo_root"])
    t0 = time.perf_counter()
    cmap = static_map(root, state["idea"], state.get("repo_id", ""))
    cmap = await enrich_map(_client(config), cmap, _understanding(state), state["idea"], root, state.get("repo_id", ""))
    data = cmap.model_dump()
    return _emit(
        {
            "type": "feasibility",
            "event": "map",
            "run_id": run_id,
            "codebase_map": data,
            "took_ms": int((time.perf_counter() - t0) * 1000),
        },
        codebase_map=data,
        usage=usage.snapshot(),
    )


async def assess_node(state: RunState, config) -> dict:
    run_id = state["run_id"]
    usage = llm.activate_usage(run_id)
    root = Path(state["repo_root"])
    understanding = _understanding(state)
    cmap = state.get("codebase_map") or {}
    from ..feasibility.schemas import CodebaseMap

    cmap_model = CodebaseMap(**cmap)
    dimensions = build_dimensions(understanding, state["idea"])
    results = await asyncio.gather(
        *(
            assess_dimension(
                _client(config), d, state["idea"], understanding, cmap_model, root, state.get("repo_id", ""),
                k=int(state.get("k", 6)),
            )
            for d in dimensions
        ),
        return_exceptions=True,
    )
    claims: list[FeasibilityClaim] = []
    for dim, res in zip(dimensions, results):
        if isinstance(res, Exception):
            logger.exception("dimension %s crashed", dim["id"])
            claim = FeasibilityClaim(
                id=f"{dim['id']}-crash",
                dimension=dim["id"],
                statement=f"Assessment failed: {res}",
                evidence=[EvidenceRef(path="")],
                confidence="low",
                provisional=True,
            )
            apply_citation_gate(root, [claim])
            claims.append(claim)
        else:
            claims.extend(res[0])
    events = [{"type": "feasibility", "event": "claim", "run_id": run_id, "claim": c.model_dump()} for c in claims]
    for evt in events:
        _write(evt)
    return {"claims": [c.model_dump() for c in claims], "usage": usage.snapshot(), "events": events}


async def evidence_gate_node(state: RunState, config) -> dict:
    root = Path(state["repo_root"])
    claims = [FeasibilityClaim(**c) for c in state.get("claims", [])]
    apply_citation_gate(root, claims)
    gate = {
        dim: {
            "supported": sum(1 for c in claims if c.dimension == dim and c.status == "supported"),
            "total": sum(1 for c in claims if c.dimension == dim),
        }
        for dim in DIMENSIONS
    }
    payload = {"type": "run", "event": "state", "phase": "evidence_gate", "gate": gate, "run_id": state["run_id"]}
    return _emit(payload, claims=[c.model_dump() for c in claims], evidence_gate=gate)


async def judge_node(state: RunState, config) -> dict:
    run_id = state["run_id"]
    usage = llm.activate_usage(run_id)
    from ..feasibility.schemas import CodebaseMap

    claims = [FeasibilityClaim(**c) for c in state.get("claims", [])]
    judgment = await judge_feasibility(
        _client(config), state["idea"], _understanding(state), CodebaseMap(**(state.get("codebase_map") or {})), claims
    )
    report = FeasibilityReport(
        run_id=run_id,
        repo_id=state.get("repo_id", ""),
        idea=state["idea"],
        understanding=_understanding(state),
        codebase_map=CodebaseMap(**(state.get("codebase_map") or {})),
        claims=claims,
        dimensions=judgment["dimensions"],
        verdict=judgment["verdict"],
        blockers=judgment["blockers"],
        alternatives=judgment["alternatives"],
        reframed_idea=judgment["reframed_idea"],
        usage=usage.snapshot(),
        created_at=time.time(),
    )
    data = report.model_dump()
    payload = {
        "type": "feasibility",
        "event": "report",
        "run_id": run_id,
        "data": data,
        "verdict": report.verdict,
    }
    return _emit(
        payload,
        report=data,
        verdict=report.verdict,
        blockers=report.blockers,
        alternatives=report.alternatives,
        reframed_idea=report.reframed_idea,
        usage=usage.snapshot(),
    )


async def reframe_node(state: RunState, config) -> dict:
    run_id = state["run_id"]
    payload = {
        "type": "run",
        "event": "interrupt",
        "kind": "reframe",
        "run_id": run_id,
        "verdict": state.get("verdict"),
        "blockers": state.get("blockers", []),
        "alternatives": state.get("alternatives", []),
        "reframed_idea": state.get("reframed_idea"),
    }
    resumed = interrupt(payload)
    if isinstance(resumed, dict):
        accept = bool(resumed.get("accept"))
        idea = resumed.get("idea")
    else:
        accept = bool(resumed)
        idea = None

    if not accept:
        return _emit(
            {"type": "run", "event": "state", "phase": "reframe", "status": "declined", "run_id": run_id},
            reframe_accepted=False,
            reframes=int(state.get("reframes", 0)) + 1,
        )
    new_idea = (idea or state.get("reframed_idea") or state["idea"]).strip()
    # Fold the accepted reframe into the understanding handshake: answers force ready.
    return _emit(
        {"type": "run", "event": "state", "phase": "reframe", "status": "accepted", "idea": new_idea, "run_id": run_id},
        idea=new_idea,
        answers=[new_idea],
        understanding=None,
        claims=[],
        codebase_map=None,
        report=None,
        verdict=None,
        reframe_accepted=True,
        reframes=int(state.get("reframes", 0)) + 1,
    )


async def _run_poc(state: RunState, config, retry: bool) -> dict:
    run_id = state["run_id"]
    usage = llm.activate_usage(run_id)
    artifact = report_to_artifact(state.get("report") or {}, state["idea"])
    if retry:
        # Second attempt carries the first failure in the spec criteria.
        sandbox = state.get("sandbox") or {}
        artifact.demo_spec.acceptance_criteria.append(f"Previous attempt failed: {str(sandbox.get('pytest_log', ''))[:300]}")
    _write({"type": "poc", "event": "start", "run_id": run_id, "retry": retry})
    artifact_data: dict | None = None
    from .instrument import UsageTrackingClient

    async for evt in code_generate(UsageTrackingClient(_client(config)), artifact):
        if evt.get("type") == "code" and evt.get("event") == "artifact":
            artifact_data = evt.get("data") or {}
    data = artifact_data or {}
    sandbox = data.get("sandbox") or {}
    poc = {
        "files": [f.get("path") for f in data.get("files", [])],
        "sandbox": sandbox,
        "coding_spec": data.get("coding_spec"),
    }
    payload = {
        "type": "poc",
        "event": "result",
        "run_id": run_id,
        "retry": retry,
        "passed": bool(sandbox.get("passed")),
        "files": poc["files"],
    }
    updates: dict = {
        "poc": poc,
        "sandbox": sandbox,
        "usage": usage.snapshot(),
    }
    if retry:
        updates["debug_attempts"] = int(state.get("debug_attempts", 0)) + 1
    return _emit(payload, **updates)


async def poc_generate_node(state: RunState, config) -> dict:
    return await _run_poc(state, config, retry=False)


async def poc_retry_node(state: RunState, config) -> dict:
    return await _run_poc(state, config, retry=True)


async def evaluate_node(state: RunState, config) -> dict:
    run_id = state["run_id"]
    usage = llm.activate_usage(run_id)
    wall_ms = int((time.time() - float(state.get("started_at", time.time()))) * 1000)
    evaluation = await evaluate_poc(
        _client(config), state.get("understanding"), state.get("sandbox") or {}, usage.snapshot(), wall_ms
    )
    data = evaluation.model_dump()
    return _emit(
        {"type": "eval", "event": "result", "run_id": run_id, "evaluation": data},
        evaluation=data,
        usage=usage.snapshot(),
    )


async def finalize_node(state: RunState, config) -> dict:
    run_id = state["run_id"]
    usage = llm.usage_for(run_id).snapshot()
    if state.get("error"):
        status = "failed"
    elif state.get("evaluation"):
        status = "done"
    elif usage["calls"] >= int(state.get("max_llm_calls", 25)):
        status = "budget_exceeded"
    else:
        status = "done"

    result = RunResult(
        run_id=run_id,
        idea=state["idea"],
        repo_id=state.get("repo_id", ""),
        status=status,
        verdict=state.get("verdict"),
        report=state.get("report"),
        evaluation=state.get("evaluation"),
        poc=state.get("poc"),
        usage=usage,
        events=state.get("events", []),
        created_at=float(state.get("started_at", time.time())),
    )
    try:
        await db.rd_run_update(run_id, {"status": status, "result": result.model_dump()})
    except Exception:
        logger.exception("rd_run_update failed for %s", run_id)
    payload = {
        "type": "run",
        "event": "result",
        "run_id": run_id,
        "status": status,
        "verdict": state.get("verdict"),
        "evaluation": state.get("evaluation"),
        "data": result.model_dump(exclude={"events"}),
    }
    return _emit(payload, status=status, usage=usage)
