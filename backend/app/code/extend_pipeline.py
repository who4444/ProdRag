"""Extend Pipeline — idea + repo -> change plan -> changes -> repo sandbox.

Streaming NDJSON (same event family as code/pipeline.py, every event tagged
mode="extend"): extend_start, context, plan, change_start|change_result,
tester/result, sandbox/start|result, artifact. Best-effort persistence into
code_runs/code_artifacts. Never raises: failures yield extend_error events.
"""

import asyncio
import logging
import time
import uuid

from ..core import db
from .extend_analyzer import analyze_extension
from .extend_orchestrator import build_extend_plan
from .extend_subagent import run_change_task
from .extend_tester import generate_extend_tests
from .repo_context import build_repo_context
from .repo_sandbox import run_repo_sandbox
from .schemas import CodeExtendRequest, ExtendArtifact, GeneratedFile, RepoContext, SandboxResult

logger = logging.getLogger(__name__)


def _norm_req(req) -> CodeExtendRequest:
    if isinstance(req, CodeExtendRequest):
        return req
    if isinstance(req, dict):
        return CodeExtendRequest(
            idea=str(req.get("idea", "")),
            repo_id=req.get("repo_id") or None,
            path=req.get("path") or None,
            entrypoint=str(req.get("entrypoint", "")),
            session_id=req.get("session_id"),
        )
    raise ValueError("request must be CodeExtendRequest or dict")


def _lite_repo(ctx: RepoContext) -> RepoContext:
    """Artifact/UI-safe repo: file texts replaced by line counts."""
    return RepoContext(
        repo_id=ctx.repo_id,
        root=ctx.root,
        mode=ctx.mode,
        files={p: f"{len(t.splitlines())} lines" for p, t in ctx.files.items()},
        hits=[],
    )


async def _fail(run_id: str, message: str):
    if run_id:
        try:
            await db.code_run_update(run_id, {"status": "failed"})
        except Exception:
            pass
    yield {"type": "code", "event": "extend_error", "mode": "extend", "run_id": run_id, "message": message}


async def code_extend_stream(client, req, session_id: str | None = None, feedback: str | None = None):
    """Yield NDJSON events for a PoC extension of a codebase. Never raises."""
    try:
        request = _norm_req(req)
    except Exception as e:
        async for evt in _fail("", f"bad request: {e}"):
            yield evt
        return
    if not request.idea.strip():
        async for evt in _fail("", "idea required"):
            yield evt
        return
    if not (request.repo_id or request.path):
        async for evt in _fail("", "repo_id or path required"):
            yield evt
        return

    run_id = str(uuid.uuid4())
    req_dump = request.model_dump()
    try:
        await db.code_run_create(run_id, None, req_dump, status="running")
    except Exception:
        logger.exception("code_run_create failed for %s", run_id)

    yield {
        "type": "code",
        "event": "extend_start",
        "mode": "extend",
        "run_id": run_id,
        "idea": request.idea[:200],
        "repo_id": request.repo_id or "",
    }

    ctx = await build_repo_context(request.idea, repo_id=request.repo_id, path=request.path, k=8)
    if ctx.mode == "none" and not ctx.files and not ctx.hits:
        async for evt in _fail(run_id, "target repo not available (index it via /repo/index or pass a local path)"):
            yield evt
        return

    yield {
        "type": "code",
        "event": "context",
        "mode": "extend",
        "run_id": run_id,
        "repo_id": ctx.repo_id,
        "root": ctx.root,
        "repo_mode": ctx.mode,
        "n_files": len(ctx.files),
        "hits": [
            {key: h.get(key) for key in ("path", "symbol", "start_line", "end_line", "score")}
            for h in ctx.hits[:8]
        ],
    }

    spec = await analyze_extension(client, request.idea, ctx)
    tasks = build_extend_plan(spec, ctx)
    if not tasks:
        async for evt in _fail(run_id, "planner produced no usable changes"):
            yield evt
        return
    yield {
        "type": "code",
        "event": "plan",
        "mode": "extend",
        "run_id": run_id,
        "extend_spec": spec.model_dump(),
        "file_tasks": [t.model_dump() for t in tasks],
    }

    sem = asyncio.Semaphore(3)

    async def _run_one(t):
        async with sem:
            return await run_change_task(client, t, ctx, feedback=feedback)

    for t in tasks:
        yield {
            "type": "code",
            "event": "change_start",
            "mode": "extend",
            "run_id": run_id,
            "path": t.path,
            "action": t.action,
            "goal": t.goal,
        }

    t0 = time.perf_counter()
    results = await asyncio.gather(*(_run_one(t) for t in tasks), return_exceptions=True)
    filled = []
    for task, res in zip(tasks, results):
        if isinstance(res, Exception):
            logger.exception("change task %s failed", task.path)
            res = task
        filled.append(res)
        yield {
            "type": "code",
            "event": "change_result",
            "mode": "extend",
            "run_id": run_id,
            "file": {
                "path": res.path,
                "action": res.action,
                "content": res.content,
                "has_search_replace": bool(res.search_replace),
            },
        }
    logger.info("extend pipeline %s: %d changes in %.2fs", run_id, len(filled), time.perf_counter() - t0)

    test_file = await generate_extend_tests(client, request.idea, spec, ctx, feedback=feedback)
    yield {"type": "tester", "event": "result", "mode": "extend", "run_id": run_id, "file": test_file.model_dump()}

    yield {"type": "sandbox", "event": "start", "mode": "extend", "run_id": run_id}
    try:
        sandbox, _applied = await run_repo_sandbox(
            ctx.root or request.path or "",
            ctx.files,
            filled,
            timeout=120,
            entrypoint=spec.entrypoint,
        )
    except Exception as e:
        logger.exception("extend sandbox crashed")
        sandbox = SandboxResult(passed=False, pytest_log=f"repo sandbox error: {e}"[:2000], returncode=1, runtime_ms=0)
    yield {"type": "sandbox", "event": "result", "mode": "extend", "run_id": run_id, "sandbox": sandbox.model_dump()}

    artifact = ExtendArtifact(
        artifact_run_id=run_id,
        repo=_lite_repo(ctx),
        extend_spec=spec,
        changes=[GeneratedFile(path=t.path, content=t.content, action=t.action) for t in filled],
        test_file=test_file,
        sandbox=sandbox,
        created_at=time.time(),
    )
    yield {
        "type": "code",
        "event": "artifact",
        "mode": "extend",
        "run_id": run_id,
        "data": artifact.model_dump(),
        "changes": [c.model_dump() for c in artifact.changes],
    }

    try:
        await db.code_artifact_upsert(run_id, artifact.model_dump(), sandbox.model_dump())
        await db.code_run_update(run_id, {"status": "done" if sandbox.passed else "failed"})
    except Exception:
        logger.exception("extend artifact persist failed for %s", run_id)
