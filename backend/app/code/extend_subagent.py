"""Extend Subagent — fill content for ExtendFileTask (create/modify).

One LLM call per task the planner left empty; modify tasks receive the current
file content and must return the COMPLETE updated file. Never raises — tasks
that fail stay content-empty and the patcher skips them safely.
"""

import logging

from ..config import settings
from .repo_context import context_digest
from .schemas import ExtendFileTask, RepoContext

logger = logging.getLogger(__name__)

CREATE_SYSTEM = (
    "You are a senior engineer extending an existing codebase. Write ONE new file "
    "that implements the requested change. It must integrate with the existing "
    "modules shown in the digest (import them instead of duplicating logic) and "
    "match the codebase's style. Python code unless the target path says otherwise. "
    "Return only the file content, no markdown fences, no explanation."
)

MODIFY_SYSTEM = (
    "You are a senior engineer modifying one file of an existing codebase. You get "
    "the current file content and the requested change. Return the COMPLETE updated "
    "file content — preserve every unrelated line exactly, apply only the requested "
    "change, and keep existing public APIs working. Return only the file content, "
    "no markdown fences, no explanation."
)


def strip_fences(content: str) -> str:
    content = (content or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return content


async def run_change_task(client, task: ExtendFileTask, ctx: RepoContext, feedback: str | None = None) -> ExtendFileTask:
    """Fill task.content for tasks the planner left empty. Never raises."""
    try:
        if task.content or task.search_replace:
            return task
        if task.action == "modify":
            current = (ctx.files.get(task.path) or "")[:6000]
            if not current:
                logger.warning("modify task %s has no source content available", task.path)
                return task
            system = MODIFY_SYSTEM
            prompt = (
                f"File to modify: {task.path}\n"
                f"Change: {task.goal}\n\n"
                f"Repository context:\n{context_digest(ctx, 2000)}\n\n"
                f"Current file content:\n{current}\n\n"
                f"Return the complete updated content for {task.path} now."
            )
        else:
            system = CREATE_SYSTEM
            prompt = (
                f"File to create: {task.path}\n"
                f"Change: {task.goal}\n\n"
                f"Repository context:\n{context_digest(ctx, 2500)}\n\n"
                f"Return only the complete content for {task.path} now."
            )
        if feedback:
            prompt += f"\n\nPrevious sandbox failure (fix this):\n{feedback[:2000]}"
        resp = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        content = strip_fences(resp.choices[0].message.content or "")
        if not content:
            return task
        task.content = content
        return task
    except Exception:
        logger.exception("change subagent failed for %s", task.path)
        return task
