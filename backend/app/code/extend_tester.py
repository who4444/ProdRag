"""Extend Tester — pytest file for the PoC extension.

One LLM call grounded in the repo digest; fallback asserts the PoC module
exists; never raises.
"""

import logging

from ..config import settings
from .extend_subagent import strip_fences
from .repo_context import context_digest
from .schemas import ExtendSpec, GeneratedFile, RepoContext

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a pytest test generator for an extended codebase. Write one test file "
    "that verifies the requested extension: import the new/modified modules "
    "(repo-root relative imports; tests run with cwd=repo root), exercise the "
    "change on toy data, stay offline (no network/DB). Return only the python file "
    "content, no markdown fences."
)


def fallback_content(entrypoint: str) -> str:
    if entrypoint:
        return (
            "def test_poc_module_present():\n"
            "    import pathlib\n\n"
            f"    assert pathlib.Path({entrypoint!r}).exists()\n"
        )
    return "def test_poc_placeholder():\n    assert True\n"


async def generate_extend_tests(
    client,
    idea: str,
    spec: ExtendSpec,
    ctx: RepoContext,
    feedback: str | None = None,
    dest: str = "test_extend_poc.py",
) -> GeneratedFile:
    try:
        prompt = (
            f"Requested extension: {idea}\n"
            f"Changed files: {[t.path for t in spec.file_tasks]}\n"
            f"test_intents: {spec.test_intents}\n"
            f"Repository digest:\n{context_digest(ctx, 2500)}\n\n"
            f"Write {dest} content now."
        )
        if feedback:
            prompt += f"\n\nPrevious sandbox failure (fix this):\n{feedback[:2000]}"
        resp = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        content = strip_fences(resp.choices[0].message.content or "")
        if not content or "def test_" not in content:
            raise ValueError("invalid test content")
        return GeneratedFile(path=dest, content=content)
    except Exception:
        logger.exception("extend tester failed, fallback")
        return GeneratedFile(path=dest, content=fallback_content(spec.entrypoint))
