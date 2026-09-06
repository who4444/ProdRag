"""Research agent: a single DeepSeek tool-calling loop.

ponytail: the whole "multi-agent" system is one loop — the orchestrator,
researcher, and (eventual) critic are roles in the system prompt, not separate
processes. kb_search is the only tool; figure refs ride along in results, and
memory is auto-injected at start / auto-saved at the end instead of extra tools
the model must remember to call. Add a real multi-process topology / LangGraph
only if single-brain prompting measurably limits quality.
"""

import json

from . import memory
from .config import settings
from .rag.retrieval import search_kb, source_items
from .tools.search import WEB_SEARCH_TOOL, format_results as format_web, web_search

SYSTEM_PROMPT = (
    "You are a research orchestrator with a knowledge base of ingested PDFs "
    "(text and figures) and access to past research. Decide how to approach the "
    "question. Call kb_search with focused sub-questions to gather evidence "
    "(multiple calls allowed). When you have enough evidence, write the final "
    "answer, citing text sources as [1], [2], ... and figures by object_key. "
    "Never invent facts not present in the search results."
)

KB_TOOL = {
    "type": "function",
    "function": {
        "name": "kb_search",
        "description": (
            "Search the knowledge base for text passages and figures relevant to a "
            "focused sub-question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Focused retrieval sub-question"},
                "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 4},
                "k_images": {"type": "integer", "minimum": 0, "maximum": 5, "default": 1},
            },
            "required": ["question"],
        },
    },
}


def _format_sources(result: dict) -> str:
    parts = [
        f"[{i}] (page {t['page']}, score {t['score']:.2f}) {t['text']}"
        for i, t in enumerate(result["text"], 1)
    ]
    parts += [
        f"[figure {j + 1}] page {img['page']} object_key={img['object_key']}"
        for j, img in enumerate(result["images"])
    ]
    return "\n\n".join(parts) or "No results."


def _format_episodes(episodes: list[dict]) -> str:
    return "\n\n".join(f"Q: {e['question']}\nA: {e['answer'][:500]}" for e in episodes)


async def _run_tool(name: str, args: dict) -> dict | list[dict]:
    if name == "web_search":
        return await web_search(args["query"], int(args.get("k", 5)))
    if name != "kb_search":
        return {"text": [], "images": []}
    text_hits, image_hits = await search_kb(
        args["question"],
        int(args.get("k", 4)),
        int(args.get("k_images", 1)),
    )
    return {"text": text_hits, "images": image_hits}


async def research(client, redis, question: str, session_id: str | None = None,
                   k: int = 4, k_images: int = 1):
    """Run the loop, yielding NDJSON events (agent/sources/text/memory)."""
    sid = await memory.conv_add(redis, session_id, "user", question)
    yield {"type": "memory", "event": "session", "session_id": sid}

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    episodes = await memory.remember_search(question)
    if episodes:
        messages.append(
            {"role": "system", "content": "Relevant past research:\n" + _format_episodes(episodes)}
        )
    messages.extend(await memory.conv_history(redis, sid))

    all_sources: list[dict] = []
    tools = [KB_TOOL]
    # add web_search only when configured — keeps token cost zero when offline
    from .tools.search import enabled as search_enabled

    if search_enabled():
        tools.append(WEB_SEARCH_TOOL)
    for _ in range(settings.agent_max_steps):
        resp = await client.chat.completions.create(
            model=settings.agent_model, messages=messages, tools=tools
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            answer_text = (msg.content or "").strip()
            if answer_text:
                yield {"type": "text", "delta": answer_text}
                await memory.conv_add(redis, sid, "assistant", answer_text)
                await memory.remember_save(sid, question, answer_text, all_sources)
                yield {"type": "memory", "event": "saved", "session_id": sid}
            return

        messages.append(
            {
                "role": msg.role,
                "content": msg.content,
                "tool_calls": [tc.model_dump(exclude_none=True) for tc in msg.tool_calls],
            }
        )
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            yield {"type": "agent", "event": "tool_call", "name": tc.function.name, "args": args}
            result = await _run_tool(tc.function.name, args)
            if tc.function.name == "web_search":
                content = format_web(result)  # type: ignore[arg-type]
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                # surface web hits as sources (kind=web) for UI
                web_items = [
                    {"kind": "web", "source": r.get("url", ""), "page": 0, "score": 0.0, "content": r.get("snippet", "")[:200], "caption": r.get("title", ""), "image_url": None}
                    for r in result  # type: ignore[union-attr]
                ]
                yield {"type": "sources", "items": web_items}
                yield {"type": "agent", "event": "tool_result", "name": tc.function.name, "summary": content[:300]}
            else:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": _format_sources(result)})  # type: ignore[arg-type]
                all_sources.extend(result["text"])  # type: ignore[union-attr]
                yield {"type": "sources", "items": source_items(result["text"], result["images"])}  # type: ignore[union-attr]
                yield {
                    "type": "agent",
                    "event": "tool_result",
                    "name": tc.function.name,
                    "summary": _format_sources(result)[:300],  # type: ignore[arg-type]
                }
    yield {"type": "text", "delta": "\n\n(stopped: reached max steps without a final answer)"}
