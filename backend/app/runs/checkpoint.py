"""Checkpointer wiring — Postgres/Supabase in prod, in-memory offline/tests.

Postgres checkpoints make runs durable: a run can be paused at an interrupt and
resumed after a process restart. Offline mode (or a missing database_url) falls
back to InMemorySaver so the API still boots with zero credentials.
"""

import logging
from contextlib import AsyncExitStack

from langgraph.checkpoint.memory import InMemorySaver

from ..config import settings

logger = logging.getLogger(__name__)

_saver = None
_stack: AsyncExitStack | None = None


def get_saver():
    return _saver if _saver is not None else InMemorySaver()


def saver_kind() -> str:
    return "postgres" if _saver is not None and not isinstance(_saver, InMemorySaver) else "memory"


async def init_checkpointer() -> None:
    """Best-effort: fall back to InMemorySaver on any Postgres problem."""
    global _saver, _stack
    if settings.offline or not settings.database_url:
        _saver = InMemorySaver()
        logger.info("checkpointer: memory (offline=%s database_url=%s)", settings.offline, bool(settings.database_url))
        return
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        _stack = AsyncExitStack()
        saver = await _stack.enter_async_context(AsyncPostgresSaver.from_conn_string(settings.database_url))
        await saver.setup()
        _saver = saver
        logger.info("checkpointer: postgres")
    except Exception:
        logger.exception("postgres checkpointer unavailable — falling back to memory")
        _saver = InMemorySaver()


async def close_checkpointer() -> None:
    global _stack
    if _stack is not None:
        try:
            await _stack.aclose()
        except Exception:
            logger.exception("checkpointer close failed")
        _stack = None
