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


async def build_plan_dynamic(client, requirements: Requirements, episodic_context: str = "") -> list[Direction]:
    """Try LLM-generated plan (3-7 dirs) validated vs allow-list, fallback to fixed 5."""
    import json
    import logging

    from ..config import settings

    logger = logging.getLogger(__name__)
    fixed = build_plan(requirements)
    try:
        sys_prompt = (
            "You are a research planner. Given requirements, generate 3-7 research directions "
            "as JSON {\"directions\": [{\"id\": str, \"question\": str, \"rationale\": str}]}. "
            "Prefer the fixed taxonomy: core_algorithm, datasets, baselines, implementation_details, evaluation, "
            "but you may add custom ids (e.g., limitations, reproducibility) if the goal warrants. "
            "Questions must be specific and retrieval-friendly. Keep rationale short."
        )
        user = f"Requirements: {requirements.model_dump()}\n\nEpisodic context: {episodic_context[:800]}\n\nReturn JSON."
        resp = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        data = json.loads((resp.choices[0].message.content or "").strip())
        dirs_in = data.get("directions") or []
        if not isinstance(dirs_in, list) or not (3 <= len(dirs_in) <= 7):
            raise ValueError(f"bad dirs len {len(dirs_in)}")
        allowed = {t["id"] for t in TAXONOMY}
        out: list[Direction] = []
        seen = set()
        for d in dirs_in:
            did = str(d.get("id", "")).strip()
            q = str(d.get("question", "")).strip()
            r = str(d.get("rationale", "")).strip()
            if not did or not q or did in seen:
                continue
            # Allow fixed ids + 1-2 custom; validate custom ids are snake_case
            if did not in allowed and not did.replace("_", "").isalnum():
                continue
            out.append(Direction(id=did, question=q, rationale=r or f"Needed for {did}"))
            seen.add(did)
        if len(out) < 3:
            raise ValueError("too few valid dirs")
        logger.info("dynamic plan: %s", [d.id for d in out])
        return out
    except Exception:
        logging.getLogger(__name__).exception("dynamic plan failed, fallback fixed")
        return fixed
