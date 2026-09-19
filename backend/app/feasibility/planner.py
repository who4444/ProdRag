"""Feasibility planner — fixed 3-dimension plan (technical | integration | data).

No LLM call: the taxonomy is the contract; the subagent writes findings only.
"""

from .schemas import IdeaUnderstanding


TAXONOMY: list[dict] = [
    {
        "id": "technical",
        "template": (
            "Can '{goal}' be implemented with the existing modules, patterns, and dependencies? "
            "What code supports or contradicts it?"
        ),
    },
    {
        "id": "integration",
        "template": (
            "Which existing modules/components does '{goal}' affect, and where is the right "
            "extension point? What breaks or must change?"
        ),
    },
    {
        "id": "data",
        "template": (
            "What data does '{goal}' need? Does it already exist in the repo/existing sources, "
            "or must it be collected? Is it clean enough to use directly?"
        ),
    },
]


def build_dimensions(understanding: IdeaUnderstanding, idea: str) -> list[dict]:
    goal = (understanding.interpretation or idea).strip()
    dims = []
    for entry in TAXONOMY:
        dims.append(
            {
                "id": entry["id"],
                "question": entry["template"].format(goal=goal[:300]),
                "rationale": f"Required to judge {entry['id']} plausibility",
            }
        )
    return dims
