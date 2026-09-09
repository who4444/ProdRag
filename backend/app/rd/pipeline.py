"""Top-level R&D pipeline — orchestrator -> subagents -> validator -> artifact.

Streams NDJSON events: orchestrator/plan, subagent/*, validator/result, artifact.
"""

import asyncio
import logging
import time
import uuid

from ..config import settings
from ..core import db
from .artifacts import build_artifact
from .orchestrator import build_plan, build_plan_dynamic
from .schemas import Requirements
from .subagent import run_direction
from .validator import validate

logger = logging.getLogger(__name__)


async def rd_research(client, requirements: Requirements, session_id: str | None, idea: str | None, k: int = 4, k_images: int = 1):
    """Yield NDJSON events for the full R&D research phase."""
    t0 = time.perf_counter()
    run_id = str(uuid.uuid4())
    # persist run early (best-effort — don't fail the stream if DB is down)
    try:
        await db.research_run_create(run_id, session_id, idea or requirements.goal, requirements.model_dump())
    except Exception:
        logger.exception("research_run_create failed for %s", run_id)

    # Memory-aware: inject past episodes into context (no extra LLM cost)
    episodic_context = ""
    try:
        from .. import memory as mem_mod

        episodes = await mem_mod.remember_search(idea or requirements.goal)
        if episodes and isinstance(episodes, list):
            episodic_context = "\n\n".join(f"Q: {e.get('question','')}\nA: {e.get('answer','')[:400]}" for e in episodes[:3] if isinstance(e, dict))
            logger.info("rd_research %s: injected %d episodes", run_id, len(episodes))
    except Exception:
        logger.exception("rd_research %s: memory injection failed", run_id)

    # Dynamic planning: try LLM-generated plan, fallback to fixed 5
    try:
        directions = await build_plan_dynamic(client, requirements, episodic_context)
        logger.info("rd_research %s: dynamic plan %d dirs", run_id, len(directions))
    except Exception:
        logger.exception("rd_plan dynamic failed, fallback fixed")
        directions = build_plan(requirements)

    yield {"type": "orchestrator", "event": "plan", "run_id": run_id, "directions": [d.model_dump() for d in directions]}
    logger.info("rd_research %s: plan %s in %.2fs", run_id, [d.id for d in directions], time.perf_counter() - t0)

    summaries: list[dict] = []
    t_sub = time.perf_counter()
    sem = asyncio.Semaphore(5)

    async def _run_one(d):
        async with sem:
            try:
                return await run_direction(client, d, k=k, k_images=k_images)
            except Exception as exc:
                logger.exception("subagent %s failed", d.id)
                from .schemas import ResearchSummary

                return ResearchSummary(direction_id=d.id, findings=f"Error: {exc}", sources=[], confidence="low", gaps=[str(exc)])

    # Emit all starts first (frontend shows searching state for all)
    for d in directions:
        yield {"type": "subagent", "event": "start", "run_id": run_id, "direction_id": d.id, "question": d.question}

    # Parallel gather — max vs sum latency
    results = await asyncio.gather(*(_run_one(d) for d in directions), return_exceptions=True)

    for d, res in zip(directions, results):
        if isinstance(res, Exception):
            logger.exception("subagent %s failed", d.id)
            from .schemas import ResearchSummary

            res = ResearchSummary(direction_id=d.id, findings=f"Error: {res}", sources=[], confidence="low", gaps=[str(res)])
        yield {"type": "subagent", "event": "result", "run_id": run_id, "summary": res.model_dump()}
        summaries.append(res.model_dump())
    logger.info("rd_research %s: subagents %d done in %.2fs", run_id, len(summaries), time.perf_counter() - t_sub)

    # coerce summaries back to objects for validator/artifact
    from .schemas import ResearchSummary

    summary_objs = [ResearchSummary(**s) for s in summaries]

    validation = await validate(client, requirements, summary_objs)
    yield {"type": "validator", "event": "result", "run_id": run_id, "validation": validation.model_dump()}

    # Validator-driven replan: if missing and small, do one extra re-search for those dirs
    if validation.missing and len(validation.missing) <= 2 and len(validation.missing) < len(directions):
        logger.info("rd_research %s: validator missing %s -> replan %d dirs", run_id, validation.missing, len(validation.missing))
        # Find missing directions or create new ones for missing ids
        missing_dirs = [d for d in directions if d.id in validation.missing]
        # If missing id not in taxonomy, create synthetic direction
        for mid in validation.missing:
            if mid not in [d.id for d in missing_dirs]:
                missing_dirs.append(__import__('backend.app.rd.schemas', fromlist=['Direction']).Direction(id=mid, question=f"What is {mid} for {requirements.goal}?", rationale="Replan for missing coverage"))
        for d in missing_dirs:
            yield {"type": "subagent", "event": "start", "run_id": run_id, "direction_id": d.id, "question": d.question + " (replan)"}
        try:
            extra_results = await asyncio.gather(*(run_direction(client, d, k=k, k_images=k_images) for d in missing_dirs), return_exceptions=True)
            for d, res in zip(missing_dirs, extra_results):
                if isinstance(res, Exception):
                    continue
                # Replace or append
                for i, s in enumerate(summaries):
                    if s["direction_id"] == d.id:
                        summaries[i] = res.model_dump()
                        summary_objs[i] = res
                        break
                else:
                    summaries.append(res.model_dump())
                    summary_objs.append(res)
                yield {"type": "subagent", "event": "result", "run_id": run_id, "summary": res.model_dump()}
            # Re-validate after replan
            validation = await validate(client, requirements, summary_objs)
            yield {"type": "validator", "event": "result", "run_id": run_id, "validation": validation.model_dump(), "replan": True}
        except Exception:
            logger.exception("replan failed")

    artifact = build_artifact(requirements, summary_objs, validation)
    yield {"type": "artifact", "data": artifact.model_dump(), "run_id": run_id}
    logger.info("rd_research %s: done in %.2fs", run_id, time.perf_counter() - t0)

    # persist artifact + mark run done (best-effort)
    try:
        await db.research_artifact_upsert(run_id, artifact.model_dump(), validation.model_dump())
        await db.research_run_update(run_id, {"status": "done"})
    except Exception:
        logger.exception("research artifact persist failed for %s", run_id)
