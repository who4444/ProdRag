"""Analyzer — turns a raw idea into Requirements, asking clarifying Qs when needed.

No KB access (cheap). Second-request handshake: first call without answers may
return ready=false with questions; client re-POSTs with answers.

ponytail: one prompt, one JSON parse, no tools, no framework.
"""

import json
import logging

from ..config import settings
from .schemas import Requirements

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are an R&D analyzer. Given a research paper idea and optionally the user's "
    "answers to prior clarifying questions, output structured requirements for a "
    "runnable demo. "
    "Rules:\n"
    "- If the idea is vague (missing core idea, demo type, constraints, or audience), "
    "ask 1-3 specific clarifying questions and set ready=false.\n"
    "- If answers were just provided, use them to fill gaps and set ready=true.\n"
    "- demo_type is script or notebook in v1 (no gradio/docker).\n"
    "- Be concise. Return JSON only.\n"
    "Output JSON shape:\n"
    '{"ready": bool, "requirements": {goal, core_idea, demo_type, constraints, audience, open_questions}, '
    '"questions": ["..."]}'
)

REQUIREMENTS_KEYS = {"goal", "core_idea", "demo_type", "constraints", "audience", "open_questions"}


def _fallback_requirements(idea: str) -> Requirements:
    return Requirements(
        goal=idea[:300],
        core_idea=idea[:300],
        demo_type="script",
        constraints=[],
        audience="researcher",
        open_questions=[],
    )


async def analyze_idea(client, idea: str, answers: list[str] | None = None) -> dict:
    """Run the analyzer LLM call. Returns {ready, requirements?, questions?, draft?}."""
    user_content = f"Idea: {idea}"
    if answers:
        user_content += "\n\nUser answers to prior questions:\n" + "\n".join(
            f"- {a}" for a in answers
        )
        user_content += "\n\nNow produce final requirements with ready=true."

    try:
        resp = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
    except Exception:
        logger.exception("analyzer failed, falling back")
        return {"ready": True, "requirements": _fallback_requirements(idea), "questions": None, "draft": None}

    ready = bool(data.get("ready", False))
    # If answers were supplied, force ready (second-request handshake: infer defaults).
    if answers and not ready:
        ready = True

    req_data = data.get("requirements") or {}
    # sanitize to expected keys
    clean = {k: req_data.get(k) for k in REQUIREMENTS_KEYS if k in req_data}
    if not clean.get("goal"):
        clean["goal"] = idea[:300]
    if not clean.get("core_idea"):
        clean["core_idea"] = idea[:300]
    if clean.get("demo_type") not in ("script", "notebook"):
        clean["demo_type"] = "script"
    clean.setdefault("constraints", [])
    clean.setdefault("audience", "researcher")
    clean.setdefault("open_questions", [])

    try:
        requirements = Requirements(**clean)
    except Exception:
        logger.warning("analyzer produced invalid requirements, falling back")
        requirements = _fallback_requirements(idea)
        ready = True

    questions = data.get("questions") if not ready else None
    if questions is not None:
        questions = [str(q) for q in questions][:3]

    draft = requirements if not ready else None
    return {
        "ready": ready,
        "requirements": requirements if ready else None,
        "questions": questions,
        "draft": draft,
    }
