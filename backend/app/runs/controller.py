"""Run controller — start/resume/state over the compiled LangGraph.

Yields the same NDJSON event vocabulary as the rest of the API. Interrupts from
the graph surface as `{"type":"run","event":"interrupt", ...}`; the client
resumes with POST /runs/{id}/resume and the checkpointer restores the state.
"""

import logging
import time
import uuid

from langgraph.types import Command

from ..core import db, llm
from ..feasibility.pipeline import repo_root
from .graph import get_graph
from .schemas import RunRequest
from .state import RunState

logger = logging.getLogger(__name__)


def _config(run_id: str, client=None) -> dict:
    configurable: dict = {"thread_id": run_id}
    if client is not None:
        configurable["client"] = client
    return {"configurable": configurable}


def _interrupt_value(chunk: dict):
    intr = chunk.get("__interrupt__")
    if not intr:
        return None
    first = intr[0]
    return getattr(first, "value", None)


async def _persist_event(run_id: str, payload: dict) -> None:
    try:
        await db.rd_event_add(run_id, payload)
    except Exception:
        logger.debug("rd_event_add failed for %s", run_id, exc_info=True)


async def _stream(client, run_id: str, initial, command=None):
    graph = get_graph()
    config = _config(run_id, client)
    result_emitted = False
    try:
        async for mode, chunk in graph.astream(
            initial if command is None else command, config, stream_mode=["custom", "updates"]
        ):
            if mode == "custom":
                await _persist_event(run_id, chunk)
                if chunk.get("type") == "run" and chunk.get("event") == "result":
                    result_emitted = True
                yield chunk
            else:
                value = _interrupt_value(chunk)
                if value is not None:
                    payload = {"type": "run", "event": "interrupt", "run_id": run_id, **value}
                    await _persist_event(run_id, payload)
                    yield payload
    except Exception as exc:
        logger.exception("run %s stream failed", run_id)
        try:
            await db.rd_run_update(run_id, {"status": "failed"})
        except Exception:
            pass
        yield {"type": "run", "event": "error", "run_id": run_id, "error": str(exc)}
    finally:
        if result_emitted:
            llm.release_usage(run_id)


async def start_run(client, req: RunRequest):
    run_id = str(uuid.uuid4())
    root = await repo_root(req.repo_id)  # ValueError -> HTTP 400 before streaming
    llm.activate_usage(run_id)
    try:
        await db.rd_run_create(run_id, req.repo_id, req.idea)
    except Exception:
        logger.exception("rd_run_create failed for %s", run_id)

    state: RunState = {
        "run_id": run_id,
        "idea": req.idea,
        "repo_id": req.repo_id,
        "repo_root": str(root),
        "k": req.k,
        "max_llm_calls": req.max_llm_calls,
        "max_wall_s": req.max_wall_s,
        "started_at": time.time(),
        "questions": [],
        "claims": [],
        "reframes": 0,
        "debug_attempts": 0,
        "status": "running",
        "events": [],
        "usage": {},
    }
    yield {
        "type": "run",
        "event": "start",
        "run_id": run_id,
        "repo_id": req.repo_id,
        "max_llm_calls": req.max_llm_calls,
        "max_wall_s": req.max_wall_s,
    }
    async for evt in _stream(client, run_id, state):
        yield evt


async def resume_run(client, run_id: str, value):
    llm.activate_usage(run_id)
    yield {"type": "run", "event": "resumed", "run_id": run_id}
    async for evt in _stream(client, run_id, None, command=Command(resume=value)):
        yield evt


async def get_run_state(run_id: str) -> dict:
    graph = get_graph()
    snapshot = await graph.aget_state(_config(run_id))
    values = dict(snapshot.values or {})
    interrupts = []
    for task in snapshot.tasks or []:
        for intr in getattr(task, "interrupts", ()) or ():
            interrupts.append(getattr(intr, "value", None))
    return {
        "run_id": run_id,
        "next": list(snapshot.next or []),
        "interrupts": interrupts,
        "values": values,
    }
