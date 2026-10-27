"""Extend Analyzer — idea + RepoContext -> ExtendSpec.

One json_object LLM call grounded in the repo digest; deterministic fallback
plans a minimal single-file change; never raises.
"""

import json
import logging

from ..config import settings
from .repo_context import context_digest
from .schemas import ExtendFileTask, ExtendSpec, RepoContext

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a code-extension planner. Given an idea and a digest of an existing "
    "codebase (file map + retrieved snippets), output an ExtendSpec JSON that "
    "realizes the idea inside that codebase.\n"
    "Shape: {\"entrypoint\": str, \"focus_paths\": [str], \"file_tasks\": "
    "[{\"path\": str, \"action\": \"create\"|\"modify\", \"goal\": str, "
    "\"content\": str, \"search_replace\": [{\"search\": str, \"replace\": str}]}], "
    "\"test_intents\": [str], \"dependencies\": [str], \"risks\": [str]}\n"
    "Rules:\n"
    "- file_tasks paths are repo-relative posix paths. action=modify only for paths "
    "that exist in the file map; new modules/config/action='create'.\n"
    "- For modify: EITHER provide full replacement 'content' OR 1-3 exact "
    "search_replace blocks (search must be a verbatim unique snippet from the file).\n"
    "- Keep changes minimal: 1-3 file_tasks. New files should integrate with existing "
    "modules (import them) rather than duplicating logic.\n"
    "- entrypoint: optional runnable module to smoke-run after tests (repo-relative, "
    "often the new/modified file); empty string if none.\n"
    "- test_intents describe how existing or new pytest files will verify the change; "
    "the pipeline validates via the repo's own pytest suite.\n"
    "- dependencies lists ONLY packages the changed code needs that are not likely "
    "already present.\n"
    "- risks note anything that could break existing behavior.\n"
    "Return JSON only."
)


def _fallback_spec(idea: str, ctx: RepoContext) -> ExtendSpec:
    # Safe fallback: always CREATE a fresh PoC module — never rewrite an
    # existing repo file from a fallback path (that would clobber real code).
    target = "extend_poc.py"
    return ExtendSpec(
        entrypoint=target,
        focus_paths=[target],
        file_tasks=[ExtendFileTask(
            path=target,
            action="create",
            goal=idea[:200],
            content=(
                f"\"\"\"PoC extension: {idea[:150]}\"\"\"\n"
                "def main() -> dict:\n"
                f"    return {{\"extended\": \"{target}\", \"idea\": {idea[:80]!r}}}\n"
                "\n"
                "if __name__ == \"__main__\":\n"
                "    import json\n"
                "    print(json.dumps(main()))\n"
            ),
        )],
        test_intents=[f"{target} runs without error", "existing repo tests still pass"],
        dependencies=[],
        risks=["fallback plan used (LLM planner unavailable)"],
    )


async def analyze_extension(client, idea: str, ctx: RepoContext, max_changes: int = 3) -> ExtendSpec:
    try:
        digest = context_digest(ctx, max_chars=6000)
        resp = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Idea: {idea}\n\nRepository digest:\n{digest}\n\nReturn ExtendSpec JSON."},
            ],
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        tasks: list[ExtendFileTask] = []
        for t in (data.get("file_tasks") or [])[:max_changes + 2]:
            if not isinstance(t, dict) or not str(t.get("path", "")).strip():
                continue
            tasks.append(ExtendFileTask(
                path=str(t.get("path", "")).strip(),
                action="modify" if str(t.get("action", "create")).lower() == "modify" else "create",
                goal=str(t.get("goal", ""))[:500],
                content=str(t.get("content", "")),
                search_replace=[b for b in (t.get("search_replace") or []) if isinstance(b, dict)][:3],
            ))
        if not tasks:
            raise ValueError("planner returned no usable file_tasks")
        if len(tasks) > max_changes:
            tasks = tasks[:max_changes]
        spec = ExtendSpec(
            entrypoint=str(data.get("entrypoint", "")),
            focus_paths=[str(p) for p in (data.get("focus_paths") or [])][:5],
            file_tasks=tasks,
            test_intents=[str(x) for x in (data.get("test_intents") or [])][:5],
            dependencies=[str(x) for x in (data.get("dependencies") or [])][:10],
            risks=[str(x) for x in (data.get("risks") or [])][:5],
        )
        # entrypoint must point at a changed file (or stay empty)
        changed = {t.path for t in tasks}
        if spec.entrypoint and spec.entrypoint not in changed:
            spec.entrypoint = tasks[0].path
        return spec
    except Exception:
        logger.exception("extend analyzer failed, falling back")
        return _fallback_spec(idea, ctx)
