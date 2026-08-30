import asyncio

import pytest

from app.core.retry import retry_async


def test_retries_until_success():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    result = asyncio.run(retry_async(flaky, base_delay=0))
    assert result == "ok"
    assert calls["n"] == 3


def test_gives_up_after_attempts():
    calls = {"n": 0}

    async def always_fails():
        calls["n"] += 1
        raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        asyncio.run(retry_async(always_fails, attempts=2, base_delay=0))
    assert calls["n"] == 2


def test_does_not_retry_nontransient():
    calls = {"n": 0}

    async def auth_error():
        calls["n"] += 1
        raise ValueError("bad config")

    with pytest.raises(ValueError):
        asyncio.run(
            retry_async(
                auth_error,
                attempts=3,
                base_delay=0,
                is_transient=lambda e: isinstance(e, ConnectionError),
            )
        )
    assert calls["n"] == 1