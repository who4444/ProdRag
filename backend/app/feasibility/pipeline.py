"""Top-level feasibility pipeline — understand -> map -> subagents -> judge -> report.

Streams NDJSON events: feasibility/start, understand, clarify, map, claim, report.
Persists to feasibility_runs/feasibility_reports best-effort (never fails the stream).
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path

from ..core import db, llm
from .codebase_map import enrich_map, static_map
from .judge import judge_feasibility
from .planner import build_dimensions
from .schemas import AnalyzeRequest, EvidenceRef, FeasibilityClaim, FeasibilityReport
from .subagent import assess_dimension
from .understand import understand_idea

logger = logging.getLogger(__name__)


async def repo_root(repo_id: str) -> Path:
    """Resolve an indexed repo id to a live directory. Raises ValueError when unusable."""
    repo = await db.repo_get(repo_id)
    if not repo:
        raise ValueError(f"repo '{repo_id}' is not indexed — POST /repo/index first")
    root = Path(repo.get("root", ""))
    if not root.is_dir():
        raise ValueError(f"indexed repo path no longer exists: {root}")
    return root


async def feasibility_analyze(client, req: AnalyzeRequest, root: Path):
    """Yield NDJSON events for the full feasibility phase."""
    t0 = time.perf_counter()
    run_id = str(uuid.uuid4())
    usage = llm.start_usage()

    # 1. understand the idea (skip when the client already completed the handshake)
    if req.understanding is not None:
        understanding = req.understanding
        ready, questions = True, None
    else:
        result = await understand_idea(client, req.idea, req.answers)
        understanding = result["understanding"] if result["ready"] else None
        ready, questions = result["ready"], result["questions"]

    yield {"type": "feasibility", "event": "start", "run_id": run_id, "repo_id": req.repo_id}
    if not ready or understanding is None:
        yield {"type": "feasibility", "event": "clarify", "run_id": run_id, "questions": questions or []}
        return
    yield {"type": "feasibility", "event": "understand", "run_id": run_id, "understanding": understanding.model_dump()}

    try:
        await db.feasibility_run_create(run_id, req.repo_id, req.idea, understanding.model_dump())
    except Exception:
        logger.exception("feasibility_run_create failed for %s", run_id)

    # 2. codebase map: deterministic first, one LLM enrichment call second
    cmap = static_map(root, req.idea, req.repo_id)
    cmap = await enrich_map(client, cmap, understanding, req.idea, root, req.repo_id)
    yield {"type": "feasibility", "event": "map", "run_id": run_id, "codebase_map": cmap.model_dump()}

    # 3. three dimension subagents in parallel
    dimensions = build_dimensions(understanding, req.idea)
    results = await asyncio.gather(
        *(
            assess_dimension(client, d, req.idea, understanding, cmap, root, req.repo_id, k=req.k)
            for d in dimensions
        ),
        return_exceptions=True,
    )
    claims: list[FeasibilityClaim] = []
    for d, res in zip(dimensions, results):
        if isinstance(res, Exception):
            logger.exception("dimension %s crashed", d["id"])
            dim_claims = [
                FeasibilityClaim(
                    id=f"{d['id']}-crash",
                    dimension=d["id"],
                    statement=f"Assessment failed: {res}",
                    evidence=[EvidenceRef(path="")],
                    confidence="low",
                    provisional=True,
                )
            ]
            from .citation import apply_citation_gate

            dim_claims = apply_citation_gate(root, dim_claims)
        else:
            dim_claims = res[0]
        for claim in dim_claims:
            claims.append(claim)
            yield {"type": "feasibility", "event": "claim", "run_id": run_id, "claim": claim.model_dump()}

    # 4. judge + deterministic evidence gate
    judged = await judge_feasibility(client, req.idea, understanding, cmap, claims)
    report = FeasibilityReport(
        run_id=run_id,
        repo_id=req.repo_id,
        idea=req.idea,
        understanding=understanding,
        codebase_map=cmap,
        claims=claims,
        dimensions=judged["dimensions"],
        verdict=judged["verdict"],
        blockers=judged["blockers"],
        alternatives=judged["alternatives"],
        reframed_idea=judged["reframed_idea"],
        usage=usage.snapshot(),
        created_at=time.time(),
    )
    yield {"type": "feasibility", "event": "report", "run_id": run_id, "data": report.model_dump()}

    try:
        await db.feasibility_report_upsert(run_id, report.model_dump())
        await db.feasibility_run_update(run_id, {"status": "done", "verdict": report.verdict})
    except Exception:
        logger.exception("feasibility report persist failed for %s", run_id)
    logger.info(
        "feasibility %s: verdict=%s claims=%d in %.2fs",
        run_id,
        report.verdict,
        len(claims),
        time.perf_counter() - t0,
    )
