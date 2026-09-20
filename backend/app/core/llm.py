"""Structured-output LLM helper with per-run token/cost accounting.

Pipeline stages call chat_json() instead of the OpenAI client directly so a run
can report cost without threading a ledger through every call. The active Usage
is a mutable object held in a ContextVar: asyncio.gather children share the same
object reference, so parallel subagents accumulate into one ledger.
"""

import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


_usage: ContextVar[Usage | None] = ContextVar("llm_usage", default=None)

# Per-run ledgers for graph executions: nodes call activate_usage(run_id) so
# every chat_json inside that node accumulates into the run's Usage object,
# across interrupt/resume cycles and parallel fan-out.
_runs: dict[str, Usage] = {}


def start_usage() -> Usage:
    """Begin a usage ledger for the current task/context."""
    usage = Usage()
    _usage.set(usage)
    return usage


def activate_usage(run_id: str) -> Usage:
    """Bind the current context to a named run ledger (created on first use)."""
    usage = _runs.setdefault(run_id, Usage())
    _usage.set(usage)
    return usage


def usage_for(run_id: str) -> Usage:
    return _runs.setdefault(run_id, Usage())


def release_usage(run_id: str) -> None:
    _runs.pop(run_id, None)


def current_usage() -> Usage | None:
    return _usage.get()


def usage_snapshot() -> dict | None:
    usage = _usage.get()
    return usage.snapshot() if usage else None


def _price(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000 * settings.llm_input_price_per_mtok
        + completion_tokens / 1_000_000 * settings.llm_output_price_per_mtok
    )


def record_usage(prompt_tokens: int, completion_tokens: int) -> None:
    usage = _usage.get()
    if usage is None:
        return
    usage.calls += 1
    usage.prompt_tokens += prompt_tokens
    usage.completion_tokens += completion_tokens
    usage.cost_usd += _price(prompt_tokens, completion_tokens)


def _extract_json(content: str) -> dict:
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.strip("`")
    start = content.find("{")
    if start == -1:
        raise ValueError("LLM response contained no JSON object")
    try:
        # raw_decode takes the first complete object and tolerates trailing text
        # (models sometimes append prose or a second object).
        obj, _ = json.JSONDecoder().raw_decode(content[start:])
    except json.JSONDecodeError:
        end = content.rfind("}")
        if end <= start:
            raise ValueError("LLM response contained no JSON object") from None
        obj = json.loads(content[start : end + 1])
    if not isinstance(obj, dict):
        raise TypeError("LLM response JSON was not an object")
    return obj


async def chat_json(client, system: str, user: str, model: str | None = None) -> dict:
    """One json_object chat call; records usage when a ledger is active."""
    resp = await client.chat.completions.create(
        model=model or settings.chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    raw = getattr(resp, "usage", None)
    if raw is not None:
        record_usage(
            int(getattr(raw, "prompt_tokens", 0) or 0),
            int(getattr(raw, "completion_tokens", 0) or 0),
        )
    return _extract_json(content)
