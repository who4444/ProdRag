"""Run schemas — request/response, evaluation report, final result."""

from typing import Literal

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    idea: str = Field(min_length=1)
    repo_id: str = Field(min_length=1)
    k: int = Field(default=6, ge=1, le=20)
    max_llm_calls: int = Field(default=25, ge=3, le=200)
    max_wall_s: int = Field(default=900, ge=30, le=7200)


class ResumeRequest(BaseModel):
    answers: list[str] | None = Field(default=None, description="Answers to clarify questions")
    accept: bool | None = Field(default=None, description="Accept the reframed idea?")
    idea: str | None = Field(default=None, description="Optional user-edited reframed idea")


class HypothesisResult(BaseModel):
    hypothesis: str
    supported: bool = False
    evidence: str = ""


class EvaluationReport(BaseModel):
    runnable: bool = False
    sandbox_passed: bool = False
    tests_passed: bool = False
    spec_fulfilled: bool = False
    hypotheses: list[HypothesisResult] = Field(default_factory=list)
    verdict: Literal["pass", "pass_with_caveats", "fail"] = "fail"
    caveats: list[str] = Field(default_factory=list)
    runtime_ms: int = 0
    wall_ms: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0


class RunResult(BaseModel):
    run_id: str
    idea: str
    repo_id: str
    status: str = "running"  # running | done | failed | budget_exceeded
    verdict: str | None = None
    report: dict | None = None
    evaluation: dict | None = None
    poc: dict | None = None
    usage: dict = Field(default_factory=dict)
    events: list[dict] = Field(default_factory=list)
    created_at: float = 0
