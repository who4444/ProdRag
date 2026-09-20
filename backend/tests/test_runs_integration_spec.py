"""Integration plan: parse, render, fallback (M3.5)."""

import json

import pytest


def _report() -> dict:
    return {
        "claims": [
            {
                "dimension": "technical",
                "status": "supported",
                "statement": "rag_search is the shared entry point.",
                "evidence": [{"path": "backend/app/tools/rag.py", "start_line": 11, "end_line": 14, "valid": True}],
            }
        ],
        "codebase_map": {
            "extension_points": [{"path": "backend/app/tools/rag.py", "symbol": "rag_search", "reason": "entry point"}]
        },
    }


def _files():
    return [
        {"path": "demo.py", "content": "print('{}')"},
        {"path": "cache.py", "content": "class LRU: ..."},
    ]


class _FakeClient:
    def __init__(self, payload):
        async def _create(**kwargs):
            return type(
                "R",
                (),
                {"choices": [type("C", (), {"message": type("M", (), {"content": json.dumps(payload)})()})()]},
            )()

        self.chat = type("C", (), {"completions": type("CC", (), {"create": staticmethod(_create)})()})()


class _FailingClient:
    def __init__(self):
        async def _create(**kwargs):
            raise RuntimeError("LLM down")

        self.chat = type("C", (), {"completions": type("CC", (), {"create": staticmethod(_create)})()})()


@pytest.mark.asyncio
async def test_build_integration_plan_parses_and_renders():
    from app.runs.integration import build_integration_plan

    payload = {
        "summary": "Apply the cache at rag_search.",
        "files": [
            {
                "path": "backend/app/tools/rag.py",
                "change_type": "modify",
                "rationale": "shared entry point",
                "sketch": "cache = LRU(100)",
            },
            {"path": "", "change_type": "modify"},  # dropped
        ],
        "steps": ["add cache", "count calls"],
        "risks": ["process-local only"],
        "verification": ["pytest -q"],
    }
    plan = await build_integration_plan(
        _FakeClient(payload), "add caching", _report(), {"chosen_method": "LRU"}, _files(), {"passed": True}
    )

    assert plan.summary == "Apply the cache at rag_search."
    assert [f.path for f in plan.files] == ["backend/app/tools/rag.py"]
    assert plan.files[0].change_type == "modify"
    assert plan.steps == ["add cache", "count calls"]
    assert plan.changes_md.startswith("# Integration plan: add caching")
    assert "backend/app/tools/rag.py" in plan.changes_md
    assert "pytest -q" in plan.changes_md
    assert "was not modified" in plan.changes_md


@pytest.mark.asyncio
async def test_build_integration_plan_falls_back_to_extension_points():
    from app.runs.integration import build_integration_plan

    plan = await build_integration_plan(
        _FailingClient(), "add caching", _report(), {"system_context": "tools/rag.py"}, _files(), {"passed": True}
    )

    assert plan.files and plan.files[0].path == "backend/app/tools/rag.py"
    assert any("manual review" in r.lower() for r in plan.risks)
    assert plan.verification
    assert plan.changes_md


@pytest.mark.asyncio
async def test_build_integration_plan_skips_llm_when_sandbox_failed():
    from app.runs.integration import build_integration_plan

    plan = await build_integration_plan(
        _FailingClient(), "add caching", _report(), {}, _files(), {"passed": False}
    )
    assert plan.files and plan.files[0].path == "backend/app/tools/rag.py"
