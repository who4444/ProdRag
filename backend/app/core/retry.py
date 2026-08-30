"""Minimal async retry with exponential backoff for transient failures."""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def retry_async(
    fn,
    *args,
    attempts: int = 3,
    base_delay: float = 1.0,
    is_transient=lambda exc: True,
    label: str = "op",
    **kwargs,
):
    """Run `fn` up to `attempts` times, backing off 1s, 2s, 4s…

    Only exceptions passing `is_transient(exc)` are retried; anything else
    (4xx auth, config errors) propagates immediately. `fn` may be async or
    return a coroutine (e.g. `asyncio.to_thread(...)`).
    """
    for attempt in range(1, attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            if attempt >= attempts or not is_transient(exc):
                raise
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                label, attempt, attempts, exc, delay,
            )
            await asyncio.sleep(delay)
    raise RuntimeError(f"unreachable: {label} retries exhausted")