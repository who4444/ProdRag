"""Top-level R&D pipeline — orchestrator -> subagents -> validator -> artifact.

Streams NDJSON events: orchestrator/plan, subagent/*, validator/result, artifact.
"""

import logging
import uuid

from ..core import db
from .artifacts import build_artifact
from .orchestrator import build_plan
from .schemas import Requirements
from .subagent import run_direction
from .validator import validate

logger = logging.getLogger(__name__)


async def rd_research(client, requirements: Requirements, session_id: str | None, idea: str | None, k: int = 4, k_images: int = 1):
    """Yield NDJSON events for the full R&D research phase."""
    run_id = str(uuid.uuid4())
    # persist run early (best-effort — don't fail the stream if DB is down)
    try:
        await db.research_run_create(run_id, session_id, idea or requirements.goal, requirements.model_dump())
    except Exception:
        logger.exception("research_run_create failed for %s", run_id)

    directions = build_plan(requirements)
    yield {"type": "orchestrator", "event": "plan", "run_id": run_id, "directions": [d.model_dump() for d in directions]}

    summaries: list[dict] = []
    for d in directions:
        yield {"type": "subagent", "event": "start", "run_id": run_id, "direction_id": d.id, "question": d.question}
        summary = await run_direction(client, d, k=k, k_images=k_images)
        yield {"type": "subagent", "event": "result", "run_id": run_id, "summary": summary.model_dump()}
        summaries.append(summary.model_dump())

    # coerce summaries back to objects for validator/artifact
    from .schemas import ResearchSummary

    summary_objs = [ResearchSummary(**s) for s in summaries]

    validation = await validate(client, requirements, summary_objs)
    yield {"type": "validator", "event": "result", "run_id": run_id, "validation": validation.model_dump()}

    artifact = build_artifact(requirements, summary_objs, validation)
    yield {"type": "artifact", "data": artifact.model_dump(), "run_id": run_id}

    # persist artifact + mark run done (best-effort)
    try:
        await db.research_artifact_upsert(run_id, artifact.model_dump(), validation.model_dump())
        await db.research_run_update(run_id, {"status": "done"})
    except Exception:
        logger.exception("research artifact persist failed for %s", run_id)
