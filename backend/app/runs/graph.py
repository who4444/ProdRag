"""LangGraph state machine — developer-defined nodes, conditional edges, cycles.

Routing is deterministic (no LLM decides the next node). The graph contributes
durability (checkpoints), human-in-the-loop interrupts at clarify/reframe,
bounded retry cycles, and budget edges.
"""

import logging

from langgraph.graph import END, START, StateGraph

from .checkpoint import get_saver
from .nodes import (
    assess_node,
    budget_ok,
    clarify_node,
    codebase_map_node,
    evaluate_node,
    evidence_gate_node,
    finalize_node,
    integration_node,
    judge_node,
    poc_generate_node,
    poc_retry_node,
    reframe_node,
    research_node,
    understand_node,
)
from .state import RunState

logger = logging.getLogger(__name__)

_graph = None


def route_after_understand(state: RunState) -> str:
    if state.get("understanding"):
        return "map"
    if state.get("questions"):
        return "clarify"
    return "finalize"


def route_after_map(state: RunState) -> str:
    return "assess" if budget_ok(state) else "finalize"


def route_after_gate(state: RunState) -> str:
    return "judge" if budget_ok(state) else "finalize"


def route_after_judge(state: RunState) -> str:
    verdict = state.get("verdict")
    reframes = int(state.get("reframes", 0))
    if verdict == "plausible":
        return "research" if budget_ok(state) else "finalize"
    if verdict in ("infeasible", "plausible_with_risks") and reframes < 1:
        return "reframe"
    if verdict == "plausible_with_risks":
        return "research" if budget_ok(state) else "finalize"
    return "finalize"


def route_after_reframe(state: RunState) -> str:
    return "understand" if state.get("reframe_accepted") else "finalize"


def route_after_poc(state: RunState) -> str:
    sandbox = state.get("sandbox") or {}
    if sandbox.get("passed"):
        return "evaluate"
    if int(state.get("debug_attempts", 0)) < 1 and budget_ok(state):
        return "retry"
    return "evaluate"


def build_graph(saver=None):
    g = StateGraph(RunState)
    g.add_node("understand", understand_node)
    g.add_node("clarify", clarify_node)
    g.add_node("codebase_map", codebase_map_node)
    g.add_node("assess", assess_node)
    g.add_node("evidence_gate", evidence_gate_node)
    g.add_node("judge", judge_node)
    g.add_node("research", research_node)
    g.add_node("reframe", reframe_node)
    g.add_node("poc_generate", poc_generate_node)
    g.add_node("poc_retry", poc_retry_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("integration_plan", integration_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "understand")
    g.add_conditional_edges(
        "understand",
        route_after_understand,
        {"clarify": "clarify", "map": "codebase_map", "finalize": "finalize"},
    )
    g.add_edge("clarify", "understand")
    g.add_conditional_edges("codebase_map", route_after_map, {"assess": "assess", "finalize": "finalize"})
    g.add_edge("assess", "evidence_gate")
    g.add_conditional_edges("evidence_gate", route_after_gate, {"judge": "judge", "finalize": "finalize"})
    g.add_conditional_edges(
        "judge", route_after_judge, {"reframe": "reframe", "research": "research", "finalize": "finalize"}
    )
    g.add_edge("research", "poc_generate")
    g.add_conditional_edges("reframe", route_after_reframe, {"understand": "understand", "finalize": "finalize"})
    g.add_conditional_edges(
        "poc_generate", route_after_poc, {"evaluate": "evaluate", "retry": "poc_retry", "finalize": "finalize"}
    )
    g.add_edge("poc_retry", "evaluate")
    g.add_edge("evaluate", "integration_plan")
    g.add_edge("integration_plan", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=saver if saver is not None else get_saver())


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def reset_graph() -> None:
    """Drop the compiled graph (tests / checkpointer swap)."""
    global _graph
    _graph = None
