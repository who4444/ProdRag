"""Orchestrator — fixed 5-direction plan from Requirements."""

from .schemas import Direction, Requirements

TAXONOMY: list[dict] = [
    {
        "id": "core_algorithm",
        "template": "What is the core algorithm or technique and its key steps for: {goal}?",
    },
    {
        "id": "datasets",
        "template": "What datasets or toy data are relevant for '{goal}' and what preprocessing is needed?",
    },
    {
        "id": "baselines",
        "template": "What baselines or alternative approaches should a demo of '{goal}' compare against?",
    },
    {
        "id": "implementation_details",
        "template": "What implementation details, hyperparameters, or tricks matter for a faithful demo of '{goal}'?",
    },
    {
        "id": "evaluation",
        "template": "How is the method for '{goal}' evaluated — metrics, acceptance checks, or toy assertions?",
    },
]


def build_plan(requirements: Requirements) -> list[Direction]:
    """Build the fixed 5-direction plan from requirements. No LLM call — ponytail."""
    goal = requirements.goal.strip() or requirements.core_idea.strip()
    directions: list[Direction] = []
    for entry in TAXONOMY:
        question = entry["template"].format(goal=goal)
        # rationale is a one-liner linking the direction to the goal
        rationale = f"Needed to scope {entry['id']} for a {requirements.demo_type} demo"
        if requirements.constraints:
            rationale += f" under constraints {', '.join(requirements.constraints)}"
        rationale += "."
        directions.append(Direction(id=entry["id"], question=question, rationale=rationale))
    return directions
