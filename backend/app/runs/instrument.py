"""Usage-tracking OpenAI client proxy.

The code pipeline (analyzer/subagents/tester) calls the OpenAI client directly
and is non-streaming. This proxy records each call's token usage into the active
run ledger so PoC generation counts toward the budget and cost report.
"""

from ..core import llm


class UsageTrackingClient:
    def __init__(self, client):
        self._client = client

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    async def create(self, *args, **kwargs):
        resp = await self._client.chat.completions.create(*args, **kwargs)
        raw = getattr(resp, "usage", None)
        if raw is not None:
            llm.record_usage(
                int(getattr(raw, "prompt_tokens", 0) or 0),
                int(getattr(raw, "completion_tokens", 0) or 0),
            )
        return resp

    def __getattr__(self, item):
        return getattr(self._client, item)
