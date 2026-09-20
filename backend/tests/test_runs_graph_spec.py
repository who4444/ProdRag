"""LangGraph run control layer: interrupts, cycles, budget, checkpoints (M3)."""

import asyncio
import json

import pytest


def _fixture_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "core.py").write_text("class Engine:\n    pass\n\n\ndef run():\n    pass\n", encoding="utf-8")
    return repo


UNDERSTAND_ASK = {
    "ready": False,
    "understanding": {"interpretation": "add cache"},
    "questions": ["Which module?"],
}
UNDERSTAND_OK = {
    "ready": True,
    "understanding": {
        "interpretation": "add caching to the engine",
        "problem": "slow",
        "expected_outcome": "faster",
        "assumptions": [],
        "hypotheses": ["cache hit rate improves"],
        "ambiguities": [],
    },
    "questions": [],
}
ENRICH = {
    "extension_points": [{"path": "pkg/core.py", "symbol": "Engine", "reason": "state owner"}],
    "data_flow": "none",
    "data_available": [],
    "data_missing": [],
    "data_cleanliness": "unknown",
}
CLAIMS = {
    "claims": [
        {
            "statement": "Engine owns state.",
            "confidence": "high",
            "evidence": [{"path": "pkg/core.py", "start_line": 1, "end_line": 2}],
        }
    ]
}
JUDGE_OK = {
    "dimensions": {
        d: {"plausible": True, "rationale": "evidence", "blockers": [], "missing_evidence": []}
        for d in ("technical", "integration", "data")
    },
    "verdict": "plausible",
    "blockers": [],
    "alternatives": [],
    "reframed_idea": None,
}
JUDGE_INFEASIBLE = {
    "dimensions": {
        "technical": {"plausible": False, "rationale": "no fit", "blockers": ["needs new subsystem"], "missing_evidence": []},
        "integration": {"plausible": False, "rationale": "touches everything", "blockers": ["breaks API"], "missing_evidence": []},
        "data": {"plausible": True, "rationale": "ok", "blockers": [], "missing_evidence": []},
    },
    "verdict": "infeasible",
    "blockers": ["needs new subsystem", "breaks API"],
    "alternatives": ["cache only the embed call"],
    "reframed_idea": "cache only the embedding call",
}
CODING_SPEC = {
    "file_tasks": [
        {"path": "demo.py", "goal": "demo", "context_slice": ""},
        {"path": "requirements.txt", "goal": "deps", "context_slice": ""},
        {"path": "test_demo.py", "goal": "tests", "context_slice": ""},
    ],
    "test_intents": ["demo exits 0"],
    "dependencies": ["numpy"],
    "risks": [],
}
EVAL_PASS = {
    "hypotheses": [{"hypothesis": "cache hit rate improves", "supported": True, "evidence": "pytest passed"}],
    "caveats": [],
    "verdict": "pass",
    "performance_summary": "demo runs fast",
    "technical_debt": [],
    "extensibility": ["fits the extension point"],
    "worthiness": "worth_it",
}
POC_PLAN = {
    "chosen_method": "wrap rag_search with an LRU keyed by normalized question text",
    "implementation_requirements": ["cache in tools/rag.py", "count embedding calls"],
    "experiment_objective": "repeated identical questions perform zero embedding calls",
    "evaluation_metrics": [{"name": "embedding_calls", "direction": "bounded", "threshold": "<=1 per unique question"}],
    "system_context": "backend/app/tools/rag.py:rag_search",
}
INTEGRATION = {
    "summary": "Apply the cache at the rag_search extension point.",
    "files": [
        {
            "path": "backend/app/tools/rag.py",
            "change_type": "modify",
            "rationale": "shared retrieval entry point",
            "sketch": "cache = LRU(maxsize=100)",
        }
    ],
    "steps": ["add the cache", "count embedding calls"],
    "risks": ["process-local cache only"],
    "verification": ["pytest -q", "run demo.py and compare JSON metrics"],
}


class _Usage:
    prompt_tokens = 10
    completion_tokens = 5


def _response(content: str):
    return type(
        "R",
        (),
        {"choices": [type("C", (), {"message": type("M", (), {"content": content})()})()], "usage": _Usage()},
    )()


class RoutingClient:
    """Fake OpenAI client that routes by system prompt / target file."""

    def __init__(self, judge=None, ask_clarify=False):
        self.ask_clarify = ask_clarify
        self.judge = judge or JUDGE_OK
        self.systems: list[str] = []

        async def _create(**kwargs):
            messages = kwargs.get("messages") or []
            system = str(messages[0].get("content", "")) if messages else ""
            user = messages[-1].get("content", "") if messages else ""
            user_text = user if isinstance(user, str) else json.dumps(user)
            self.systems.append(system[:40])
            if "idea clarifier" in system:
                payload = UNDERSTAND_ASK if self.ask_clarify else UNDERSTAND_OK
                return _response(json.dumps(payload))
            if "codebase analyst" in system:
                return _response(json.dumps(ENRICH))
            if "feasibility analyst" in system:
                return _response(json.dumps(CLAIMS))
            if "feasibility judge" in system:
                return _response(json.dumps(self.judge))
            if "experiment planner" in system:
                return _response(json.dumps(POC_PLAN))
            if "evaluation analyst" in system:
                return _response(json.dumps(EVAL_PASS))
            if "integration planner" in system:
                return _response(json.dumps(INTEGRATION))
            if "code analyzer" in system:
                return _response(json.dumps(CODING_SPEC))
            if "test generator" in system:
                return _response("def test_demo():\n    assert True\n")
            if user_text.startswith("File to generate: "):
                target = user_text.splitlines()[0].removeprefix("File to generate: ").strip()
                content = {
                    "requirements.txt": "numpy\n",
                    "test_demo.py": "def test_demo():\n    assert True\n",
                }.get(target, "print('demo ok')\n")
                return _response(content)
            raise AssertionError(f"unrouted prompt: {system[:60]!r} / {user_text[:60]!r}")

        self.chat = type("C", (), {"completions": type("CC", (), {"create": staticmethod(_create)})()})()


@pytest.fixture
def graph_env(monkeypatch, tmp_path):
    from app.config import settings

    monkeypatch.setattr(settings, "offline", True)

    from app.runs import checkpoint, graph

    asyncio.run(checkpoint.init_checkpointer())
    graph.reset_graph()
    return _fixture_repo(tmp_path)


async def _index(repo, monkeypatch):
    from unittest.mock import AsyncMock

    from app.repo.indexer import index_repo

    monkeypatch.setattr("app.repo.indexer.vectorstore.ensure_collections", AsyncMock())
    monkeypatch.setattr("app.repo.indexer.vectorstore.delete_code_by_repo", AsyncMock())
    monkeypatch.setattr("app.repo.indexer.vectorstore.upsert_code", AsyncMock(return_value=1))

    async def _embed(texts):
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr("app.repo.indexer.text_emb.embed_texts", _embed)
    # offline db fallback keeps the repos row in memory, so repo_root can resolve it
    return await index_repo(str(repo))


async def _events(client, req):
    from app.runs.controller import start_run

    return [evt async for evt in start_run(client, req)]


async def _resume(client, run_id, value):
    from app.runs.controller import resume_run

    return [evt async for evt in resume_run(client, run_id, value)]


@pytest.mark.asyncio
async def test_graph_happy_path_with_clarify_interrupt_and_resume(graph_env, monkeypatch):
    from app.runs.schemas import RunRequest

    repo_id = (await _index(graph_env, monkeypatch)).repo_id
    client = RoutingClient(ask_clarify=True)

    first = await _events(client, RunRequest(idea="add caching to the engine", repo_id=repo_id))
    interrupt = next(e for e in first if e.get("type") == "run" and e.get("event") == "interrupt")
    assert interrupt["kind"] == "clarify"
    assert interrupt["questions"]
    assert not any(e.get("event") == "result" for e in first)

    client.ask_clarify = False
    second = await _resume(client, interrupt["run_id"], ["the retrieval layer"])

    kinds = [f"{e.get('type')}/{e.get('event')}" for e in second]
    assert "feasibility/understand" in kinds
    assert "feasibility/map" in kinds
    assert kinds.count("feasibility/claim") == 3
    assert "feasibility/report" in kinds
    assert "research/plan" in kinds  # lean research agent before the PoC
    assert "poc/result" in kinds
    assert "eval/result" in kinds
    assert "integration/plan" in kinds
    final = next(e for e in second if e.get("type") == "run" and e.get("event") == "result")
    assert final["status"] == "done"
    assert final["verdict"] == "plausible"
    assert final["evaluation"]["verdict"] == "pass"
    assert final["data"]["poc_plan"]["chosen_method"]
    integration = final["data"]["integration"]
    assert integration["files"][0]["path"] == "backend/app/tools/rag.py"
    assert integration["changes_md"].startswith("# Integration plan")
    assert final["data"]["usage"]["calls"] > 0


@pytest.mark.asyncio
async def test_graph_budget_edge_stops_run(graph_env, monkeypatch):
    from app.runs.schemas import RunRequest

    repo_id = (await _index(graph_env, monkeypatch)).repo_id
    events = await _events(RoutingClient(), RunRequest(idea="add caching", repo_id=repo_id, max_llm_calls=3))

    final = next(e for e in events if e.get("type") == "run" and e.get("event") == "result")
    assert final["status"] == "budget_exceeded"
    assert final["data"]["evaluation"] is None
    assert not any(e.get("event") == "report" for e in events)


@pytest.mark.asyncio
async def test_graph_reframe_interrupt_declined_keeps_infeasible_verdict(graph_env, monkeypatch):
    from app.runs.schemas import RunRequest

    repo_id = (await _index(graph_env, monkeypatch)).repo_id
    client = RoutingClient(judge=JUDGE_INFEASIBLE)

    first = await _events(client, RunRequest(idea="rewrite everything", repo_id=repo_id))
    interrupt = next(e for e in first if e.get("type") == "run" and e.get("event") == "interrupt")
    assert interrupt["kind"] == "reframe"
    assert interrupt["reframed_idea"]

    second = await _resume(client, interrupt["run_id"], {"accept": False})
    final = next(e for e in second if e.get("type") == "run" and e.get("event") == "result")
    assert final["status"] == "done"
    assert final["verdict"] == "infeasible"
    assert final["data"]["poc"] is None


@pytest.mark.asyncio
async def test_graph_reframe_accepted_loops_once_then_succeeds(graph_env, monkeypatch):
    from app.runs.schemas import RunRequest

    repo_id = (await _index(graph_env, monkeypatch)).repo_id
    client = RoutingClient(judge=JUDGE_INFEASIBLE)

    first = await _events(client, RunRequest(idea="rewrite everything", repo_id=repo_id))
    interrupt = next(e for e in first if e.get("event") == "interrupt")

    client.judge = JUDGE_OK  # second pass is feasible after reframe
    second = await _resume(client, interrupt["run_id"], {"accept": True, "idea": "cache only the embed call"})
    final = next(e for e in second if e.get("type") == "run" and e.get("event") == "result")
    assert final["status"] == "done"
    assert final["verdict"] == "plausible"
    assert final["data"]["idea"] == "cache only the embed call"
    assert any(e.get("event") == "interrupt" and e.get("kind") == "reframe" for e in first)


@pytest.mark.asyncio
async def test_graph_poc_debug_retry_once(graph_env, monkeypatch):
    from app.runs.schemas import RunRequest

    repo_id = (await _index(graph_env, monkeypatch)).repo_id
    attempts = {"n": 0, "feedback": []}

    async def _fake_code_generate(client, artifact, session_id=None, feedback=None):
        attempts["n"] += 1
        attempts["feedback"].append(feedback)
        passed = attempts["n"] > 1
        yield {"type": "code", "event": "file_result", "file": {"path": "demo.py", "content": "print(1)"}}
        yield {
            "type": "code",
            "event": "artifact",
            "data": {
                "files": [{"path": "demo.py", "content": "print(1)"}],
                "sandbox": {"passed": passed, "pytest_log": "1 failed" if not passed else "1 passed", "demo_log": "", "returncode": 0},
                "coding_spec": {},
            },
        }

    monkeypatch.setattr("app.runs.nodes.code_generate", _fake_code_generate)
    events = await _events(RoutingClient(), RunRequest(idea="add caching", repo_id=repo_id))

    poc_results = [e for e in events if e.get("type") == "poc" and e.get("event") == "result"]
    assert len(poc_results) == 2
    assert [r["retry"] for r in poc_results] == [False, True]
    # the retry must carry the first failure into the code pipeline
    assert attempts["feedback"][0] is None
    assert attempts["feedback"][1] and "1 failed" in attempts["feedback"][1]
    final = next(e for e in events if e.get("type") == "run" and e.get("event") == "result")
    assert final["status"] == "done"
    assert final["data"]["poc"]["sandbox"]["passed"] is True


@pytest.mark.asyncio
async def test_graph_skips_integration_when_poc_fails(graph_env, monkeypatch):
    from app.runs.schemas import RunRequest

    repo_id = (await _index(graph_env, monkeypatch)).repo_id

    async def _failing_code_generate(client, artifact, session_id=None, feedback=None):
        yield {
            "type": "code",
            "event": "artifact",
            "data": {
                "files": [{"path": "demo.py", "content": "raise SystemExit(1)"}],
                "sandbox": {"passed": False, "pytest_log": "1 failed", "demo_log": "boom", "returncode": 1},
                "coding_spec": {},
            },
        }

    monkeypatch.setattr("app.runs.nodes.code_generate", _failing_code_generate)
    events = await _events(RoutingClient(), RunRequest(idea="add caching", repo_id=repo_id))

    skipped = [e for e in events if e.get("type") == "integration" and e.get("event") == "skipped"]
    assert skipped and skipped[0]["reason"] == "poc_failed"
    final = next(e for e in events if e.get("type") == "run" and e.get("event") == "result")
    assert final["data"]["integration"] is None
    assert final["evaluation"]["verdict"] == "fail"


@pytest.mark.asyncio
async def test_checkpoint_survives_graph_rebuild(graph_env, monkeypatch):
    """A new compiled graph over the same saver resumes a paused run (restart proxy)."""
    from app.runs import checkpoint, graph
    from app.runs.schemas import RunRequest

    repo_id = (await _index(graph_env, monkeypatch)).repo_id
    client = RoutingClient(ask_clarify=True)
    first = await _events(client, RunRequest(idea="add caching", repo_id=repo_id))
    interrupt = next(e for e in first if e.get("event") == "interrupt")

    graph.reset_graph()  # simulate process restart: recompile against the same saver
    client.ask_clarify = False
    second = await _resume(client, interrupt["run_id"], ["the retrieval layer"])
    final = next(e for e in second if e.get("type") == "run" and e.get("event") == "result")
    assert final["status"] == "done"
    assert checkpoint.saver_kind() == "memory"
