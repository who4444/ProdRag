"""Spec-driven tests for Tester — design: from CodingSpec/acceptance_criteria, not from code.

One LLM call, fallback template, never raises.
"""

import pytest


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str):
        self._content = content

    async def create(self, **kwargs):
        return _FakeResp(self._content)


class _FakeClient:
    def __init__(self, content: str = "def test_demo():\n    assert True"):
        self.chat = type("C", (), {"completions": _FakeCompletions(content)})()


def _spec():
    try:
        from app.code.schemas import CodingSpec

        return CodingSpec(
            file_tasks=[{"path": "demo.py", "goal": "demo", "context_slice": ""}],
            test_intents=["demo.py exits 0", "output shape correct"],
            dependencies=["numpy"],
            risks=[],
        )
    except Exception:
        return {"file_tasks": [], "test_intents": ["demo.py exits 0"], "dependencies": [], "risks": []}


def _artifact_with_criteria():
    from app.rd.schemas import DemoSpec, Requirements, ResearchArtifact, ResearchSummary, Validation

    return ResearchArtifact(
        requirements=Requirements(goal="g", core_idea="c"),
        summaries=[ResearchSummary(direction_id="core_algorithm", findings="algo", sources=[], confidence="high")],
        validation=Validation(feasible=True),
        demo_spec=DemoSpec(entrypoint="demo.py", acceptance_criteria=["demo.py exits 0 on toy data", "toy metric > 0"]),
        sources=[],
        created_at=0,
    )


@pytest.mark.asyncio
async def test_tester_returns_test_file():
    """Must return GeneratedFile for test_demo.py. — design §4"""
    from app.code.tester import generate_tests

    client = _FakeClient("def test_demo_exits():\n    import subprocess; assert subprocess.run(['python','demo.py']).returncode==0")
    # tester signature: generate_tests(client, coding_spec, artifact) — accept either order
    try:
        result = await generate_tests(client, _spec(), _artifact_with_criteria())
    except TypeError:
        result = await generate_tests(client, _artifact_with_criteria(), _spec())

    dump = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    assert dump["path"] == "test_demo.py"
    assert "def test_" in dump["content"] or "assert" in dump["content"]


@pytest.mark.asyncio
async def test_tester_from_coding_spec_not_code():
    """Tester must be driven by CodingSpec.test_intents / acceptance_criteria, not by demo.py content. — design §4 critical"""
    from app.code.tester import generate_tests

    captured = {}

    class _CapCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResp("def test_from_spec():\n    assert True")

    client = type("C", (), {"chat": type("C2", (), {"completions": _CapCompletions()})()})()
    spec = _spec()
    artifact = _artifact_with_criteria()
    try:
        await generate_tests(client, spec, artifact)
    except TypeError:
        await generate_tests(client, artifact, spec)

    prompt = json_dump(captured)
    # prompt must mention test_intents or acceptance_criteria, not demo.py code
    assert "demo.py exits 0" in prompt or "toy metric" in prompt or "test_intents" in prompt.lower()


def json_dump(obj):
    import json

    try:
        return json.dumps(obj)
    except Exception:
        return str(obj)


@pytest.mark.asyncio
async def test_tester_single_llm_call():
    """Must be one LLM call. — design ponytail"""
    from app.code.tester import generate_tests

    class _CountingCompletions:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return _FakeResp("def test_x():\n    assert True")

    completions = _CountingCompletions()
    client = type("C", (), {"chat": type("C2", (), {"completions": completions})()})()
    try:
        await generate_tests(client, _spec(), _artifact_with_criteria())
    except TypeError:
        await generate_tests(client, _artifact_with_criteria(), _spec())
    assert len(completions.calls) == 1


@pytest.mark.asyncio
async def test_tester_fallback_on_failure():
    """Must return fallback template if LLM fails. — design emit-with-risks"""
    from app.code.tester import generate_tests

    class _FailingCompletions:
        async def create(self, **kwargs):
            raise RuntimeError("LLM down")

    client = type("C", (), {"chat": type("C2", (), {"completions": _FailingCompletions()})()})()
    try:
        result = await generate_tests(client, _spec(), _artifact_with_criteria())
    except TypeError:
        result = await generate_tests(client, _artifact_with_criteria(), _spec())
    dump = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    assert dump["path"] == "test_demo.py"
    assert "def test_" in dump["content"]


@pytest.mark.asyncio
async def test_tester_content_is_pytest():
    """Generated test must be valid pytest (contains def test_ or assert). — design §4"""
    from app.code.tester import generate_tests

    client = _FakeClient("def test_demo_runs():\n    import subprocess\n    assert subprocess.run(['python','demo.py']).returncode==0\n")
    try:
        result = await generate_tests(client, _spec(), _artifact_with_criteria())
    except TypeError:
        result = await generate_tests(client, _artifact_with_criteria(), _spec())
    content = result.model_dump()["content"] if hasattr(result, "model_dump") else result["content"]
    assert "def test_" in content
    assert "assert" in content
