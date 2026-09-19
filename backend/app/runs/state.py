"""Graph state — one dict threaded through every node, checkpointed by LangGraph."""

from operator import add
from typing import Annotated, TypedDict


class RunState(TypedDict, total=False):
    run_id: str
    idea: str
    repo_id: str
    repo_root: str
    k: int
    max_llm_calls: int
    max_wall_s: int
    started_at: float

    # understanding / clarify
    understanding: dict | None
    questions: list[str]
    answers: list[str]

    # feasibility
    codebase_map: dict | None
    claims: list[dict]  # replaceable (not reducer) so reframe can reset
    evidence_gate: dict
    report: dict | None
    verdict: str | None
    blockers: list[str]
    alternatives: list[str]
    reframed_idea: str | None
    reframe_accepted: bool
    reframes: int

    # poc + evaluation
    poc: dict | None
    sandbox: dict | None
    debug_attempts: int
    evaluation: dict | None

    # control
    status: str  # running | clarifying | reframing | done | failed | budget_exceeded
    error: str | None
    usage: dict
    events: Annotated[list[dict], add]
