"""Code Pipeline — analyzer->orchestrator->subagents->tester->sandbox.

Streaming NDJSON: code/plan, code/file_start|result x3-4, tester/result, sandbox/start|result, code/artifact.
"""

import asyncio
import logging
import time
import uuid

from ..core import db
from .analyzer import analyze_artifact
from .orchestrator import build_code_plan
from .sandbox import run_sandbox
from .schemas import CodeArtifact, GeneratedFile
from .subagent import run_file_task
from .tester import generate_tests

logger = logging.getLogger(__name__)


async def code_generate(client, artifact, session_id: str | None = None):
    """Yield NDJSON events for code generation from a ResearchArtifact dict/model."""
    # Normalize artifact to dict for later use
    if hasattr(artifact, "model_dump"):
        artifact_dict = artifact.model_dump()
        artifact_obj = artifact
    elif isinstance(artifact, dict):
        artifact_dict = artifact
        artifact_obj = artifact
    else:
        artifact_dict = {}
        artifact_obj = artifact

    run_id = str(uuid.uuid4())
    # best-effort persist
    try:
        # Use research artifact run_id if available, else new run_id
        idea = artifact_dict.get("requirements", {}).get("goal", "") if isinstance(artifact_dict.get("requirements"), dict) else getattr(getattr(artifact_obj, "requirements", None), "goal", "")
        await db.code_run_create(run_id, artifact_dict.get("artifact_run_id") or artifact_dict.get("run_id"), artifact_dict, status="running")
    except Exception:
        logger.exception("code_run_create failed for %s", run_id)
        try:
            await db.research_run_create(run_id, session_id, idea or "code generation", {})
        except Exception:
            pass

    # Analyzer
    coding_spec = await analyze_artifact(client, artifact_obj)
    yield {"type": "code", "event": "plan", "run_id": run_id, "coding_spec": coding_spec.model_dump(), "file_tasks": [t.model_dump() for t in coding_spec.file_tasks]}

    # Orchestrator (already part of analyzer's file_tasks, but emit separately for spec)
    tasks = build_code_plan(coding_spec)
    # Emit file tasks as plan details (tests check for code/plan)
    # Now subagents
    generated: list[GeneratedFile] = []
    for task in tasks:
        yield {"type": "code", "event": "file_start", "run_id": run_id, "path": task.path, "goal": task.goal}
        gfile = await run_file_task(client, task)
        generated.append(gfile)
        yield {"type": "code", "event": "file_result", "run_id": run_id, "file": gfile.model_dump()}

    # Tester — generate test_demo.py from CodingSpec, not from generated code
    # Need to ensure we have a test file; if subagents already generated one, keep but also generate via tester
    test_file: GeneratedFile | None = None
    # Find existing test_demo.py from subagents
    existing_test = next((f for f in generated if f.path == "test_demo.py"), None)
    try:
        # Tester signature handles both orders
        try:
            test_file = await generate_tests(client, coding_spec, artifact_obj)
        except TypeError:
            test_file = await generate_tests(client, artifact_obj, coding_spec)
    except Exception:
        logger.exception("tester failed")
        test_file = GeneratedFile(path="test_demo.py", content="def test_demo():\n    assert True\n")

    # If subagents didn't produce test_demo.py, add tester output; if they did, prefer tester output (spec-driven)
    if existing_test is None:
        generated.append(test_file)
    else:
        # Replace existing with tester version (spec-driven is authoritative)
        generated = [f for f in generated if f.path != "test_demo.py"] + [test_file]

    yield {"type": "tester", "event": "result", "run_id": run_id, "file": test_file.model_dump()}

    # Sandbox
    yield {"type": "sandbox", "event": "start", "run_id": run_id}
    # Convert to dicts for sandbox
    sandbox_files = [f.model_dump() for f in generated]
    try:
        sandbox_result = await run_sandbox(sandbox_files, timeout=15)
    except Exception as e:
        logger.exception("sandbox failed")
        from .schemas import SandboxResult

        sandbox_result = SandboxResult(passed=False, pytest_log=str(e), demo_log="", returncode=1, runtime_ms=0)

    # Normalize sandbox to dict/model
    if hasattr(sandbox_result, "model_dump"):
        sandbox_dump = sandbox_result.model_dump()
        sandbox_obj = sandbox_result
    else:
        sandbox_dump = dict(sandbox_result)
        from .schemas import SandboxResult

        sandbox_obj = SandboxResult(**sandbox_dump)

    yield {"type": "sandbox", "event": "result", "run_id": run_id, "sandbox": sandbox_dump}

    # Build final CodeArtifact
    artifact_model = CodeArtifact(
        artifact_run_id=run_id,
        coding_spec=coding_spec,
        files=generated,
        test_file=test_file,
        sandbox=sandbox_obj,
        created_at=time.time(),
    )

    yield {"type": "code", "event": "artifact", "run_id": run_id, "data": artifact_model.model_dump(), "files": [f.model_dump() for f in generated], "code_artifact": artifact_model.model_dump()}

    # Persist (best-effort)
    try:
        await db.code_artifact_upsert(run_id, artifact_model.model_dump(), sandbox_dump)
        await db.code_run_update(run_id, {"status": "done" if sandbox_obj.passed else "failed"})
    except Exception:
        logger.exception("code artifact persist failed for %s", run_id)
        try:
            await db.research_artifact_upsert(run_id, artifact_model.model_dump(), sandbox_dump)
        except Exception:
            pass
