"""core/llm.py: structured output + per-run usage ledger (M0)."""

import asyncio

import pytest


class _Usage:
    def __init__(self, prompt: int, completion: int):
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _Msg:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content: str, prompt: int = 10, completion: int = 5):
        self.choices = [_Choice(content)]
        self.usage = _Usage(prompt, completion)


class _FakeClient:
    def __init__(self, content: str):
        self.calls: list[dict] = []

        async def _create(**kwargs):
            self.calls.append(kwargs)
            return _Resp(content)

        self.chat = type("C", (), {"completions": type("CC", (), {"create": staticmethod(_create)})()})()


@pytest.mark.asyncio
async def test_chat_json_parses_fenced_json_and_records_usage(monkeypatch):
    from app.config import settings
    from app.core import llm

    monkeypatch.setattr(settings, "llm_input_price_per_mtok", 1.0)
    monkeypatch.setattr(settings, "llm_output_price_per_mtok", 2.0)

    usage = llm.start_usage()
    client = _FakeClient('```json\n{"verdict": "plausible"}\n```')
    data = await llm.chat_json(client, "system", "user")

    assert data == {"verdict": "plausible"}
    snap = usage.snapshot()
    assert snap == {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": round(10e-6 + 10e-6, 6)}
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_chat_json_tolerates_trailing_text_and_second_object(monkeypatch):
    from app.core import llm

    client = _FakeClient('{"a": 1}{"b": 2}')
    assert asyncio.run(llm.chat_json(client, "s", "u")) == {"a": 1}

    client = _FakeClient('{"ok": true}\n\nHope this helps!')
    assert asyncio.run(llm.chat_json(client, "s", "u")) == {"ok": True}


def test_chat_json_without_ledger_is_noop(monkeypatch):
    from app.config import settings
    from app.core import llm

    monkeypatch.setattr(settings, "llm_input_price_per_mtok", 1.0)
    monkeypatch.setattr(settings, "llm_output_price_per_mtok", 2.0)

    client = _FakeClient('{"a": 1}')
    data = asyncio.run(llm.chat_json(client, "s", "u"))
    assert data == {"a": 1}
    assert llm.usage_snapshot() is None


@pytest.mark.asyncio
async def test_usage_accumulates_across_gather(monkeypatch):
    """Parallel subagents share the mutable Usage object -> one ledger."""
    from app.core import llm

    usage = llm.start_usage()
    client = _FakeClient('{"ok": true}')

    async def _worker():
        await llm.chat_json(client, "s", "u")

    await asyncio.gather(_worker(), _worker(), _worker())
    snap = usage.snapshot()
    assert snap["calls"] == 3
    assert snap["prompt_tokens"] == 30
    assert snap["completion_tokens"] == 15
