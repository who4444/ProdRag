"""Idea understanding — interpretation, assumptions, hypotheses, ambiguities.

Stateless second-request handshake (same shape as rd/analyzer): the first call
may return ready=false with 1-3 questions; the client re-POSTs with answers and
ready is forced true.

The point is to expose what the system *thinks* the user wants before spending
resources judging feasibility.
"""

import logging

from ..core.llm import chat_json
from .schemas import IdeaUnderstanding

logger = logging.getLogger(__name__)

MAX_QUESTIONS = 3

SYSTEM = (
    "You are an idea clarifier for a software R&D feasibility system. Given an idea "
    "(and optionally the user's answers to prior questions), output JSON:\n"
    '{"ready": bool, "understanding": {"interpretation": str, "problem": str, '
    '"expected_outcome": str, "assumptions": [str], "hypotheses": [str], '
    '"ambiguities": [str]}, "questions": [str]}\n'
    "Rules:\n"
    "- interpretation = what the user is actually asking to build, in one or two sentences.\n"
    "- assumptions = things you are assuming but the user did not state.\n"
    "- hypotheses = falsifiable claims the PoC should test.\n"
    "- ambiguities = what is still unclear.\n"
    "- If scope/target/success-criterion is unclear, ask 1-3 specific questions and set ready=false.\n"
    "- If answers were provided, fold them in and set ready=true with empty ambiguities "
    "you can now resolve.\n"
    "Be concise. Return JSON only."
)


def _fallback(idea: str) -> IdeaUnderstanding:
    return IdeaUnderstanding(
        interpretation=idea[:300],
        problem="",
        expected_outcome="",
        assumptions=[],
        hypotheses=[],
        ambiguities=[],
    )


async def understand_idea(client, idea: str, answers: list[str] | None = None) -> dict:
    """Returns {ready, understanding?, questions?} — never raises."""
    user = f"Idea: {idea}"
    if answers:
        user += "\n\nUser answers to prior questions:\n" + "\n".join(f"- {a}" for a in answers)
        user += "\n\nFold the answers in and set ready=true."

    try:
        data = await chat_json(client, SYSTEM, user)
    except Exception:
        logger.exception("understand failed, falling back to ready=true")
        return {"ready": True, "understanding": _fallback(idea), "questions": None}

    ready = bool(data.get("ready", False))
    if answers and not ready:
        ready = True

    raw = data.get("understanding") or {}
    understanding = IdeaUnderstanding(
        interpretation=str(raw.get("interpretation") or idea)[:500],
        problem=str(raw.get("problem", ""))[:500],
        expected_outcome=str(raw.get("expected_outcome", ""))[:500],
        assumptions=[str(x)[:300] for x in (raw.get("assumptions") or [])][:6],
        hypotheses=[str(x)[:300] for x in (raw.get("hypotheses") or [])][:6],
        ambiguities=[str(x)[:300] for x in (raw.get("ambiguities") or [])][:6],
    )
    questions = None
    if not ready:
        questions = [str(q)[:300] for q in (data.get("questions") or [])][:MAX_QUESTIONS]
        if not questions:
            questions = ["What is the scope of the change, and how will you know it worked?"]
    return {"ready": ready, "understanding": understanding if ready else None, "questions": questions}
