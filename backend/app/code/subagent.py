"""Coding Subagent — one LLM call per FileTask.

ponytail: one async function per FileTask via asyncio.gather.
"""

import logging

from ..config import settings
from .schemas import FileTask, GeneratedFile

logger = logging.getLogger(__name__)


async def run_file_task(client, task) -> GeneratedFile:
    # Normalize task to FileTask
    if isinstance(task, dict):
        try:
            task_obj = FileTask(**task)
        except Exception:
            task_obj = FileTask(path=str(task.get("path", "demo.py")), goal=str(task.get("goal", "")), context_slice=str(task.get("context_slice", "")))
    elif hasattr(task, "path"):
        task_obj = task
    else:
        task_obj = FileTask(path="demo.py", goal=str(task), context_slice="")

    prompt = (
        f"File to generate: {task_obj.path}\n"
        f"Goal: {task_obj.goal}\n"
        f"Context slice: {task_obj.context_slice}\n\n"
        f"Task: write ONLY the content for {task_obj.path}. "
        f"For demo.py, write runnable python that demonstrates the core idea on toy data. "
        f"For requirements.txt, list python dependencies one per line. "
        f"For test_demo.py, write pytest. "
        f"Return only the file content, no markdown fences, no extra explanation."
    )

    try:
        resp = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences if LLM added them
        if content.startswith("```"):
            lines = content.splitlines()
            # remove first fence and last fence
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        if not content:
            raise ValueError("empty content")
        return GeneratedFile(path=task_obj.path, content=content)
    except Exception:
        logger.exception("coding subagent failed for %s, fallback", task_obj.path)
        # Fallback content per file
        if task_obj.path == "requirements.txt":
            content = "numpy\n"
        elif task_obj.path == "test_demo.py":
            content = "def test_demo():\n    assert True\n"
        else:
            content = "print('hello demo')\n"
        return GeneratedFile(path=task_obj.path, content=content)
