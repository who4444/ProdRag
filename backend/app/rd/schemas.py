"""R&D pipeline schemas — Requirements, Directions, Summaries, Artifact."""

from pydantic import BaseModel, Field


class Requirements(BaseModel):
    goal: str = Field(description="What the demo must do")
    core_idea: str = Field(description="Paper's core idea in 1-2 sentences")
    demo_type: str = Field(default="script", description="script | notebook (v1)")
    constraints: list[str] = Field(default_factory=list)
    audience: str = Field(default="researcher")
    open_questions: list[str] = Field(default_factory=list)


class Direction(BaseModel):
    id: str
    question: str
    rationale: str = ""


class ResearchSummary(BaseModel):
    direction_id: str
    findings: str
    sources: list[dict] = Field(default_factory=list)
    confidence: str = Field(default="low", description="low | med | high")
    gaps: list[str] = Field(default_factory=list)


class Validation(BaseModel):
    coverage: dict[str, bool] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    feasible: bool = True
    risks: list[str] = Field(default_factory=list)


class DemoSpec(BaseModel):
    entrypoint: str = "demo.py"
    tech_stack: list[str] = Field(default_factory=lambda: ["python", "numpy"])
    core_algorithm: str = ""
    pseudocode: str = ""
    acceptance_criteria: list[str] = Field(default_factory=list)


class ResearchArtifact(BaseModel):
    requirements: Requirements
    summaries: list[ResearchSummary] = Field(default_factory=list)
    validation: Validation = Field(default_factory=Validation)
    demo_spec: DemoSpec = Field(default_factory=DemoSpec)
    sources: list[dict] = Field(default_factory=list)
    created_at: float = 0


# API request/response schemas

class AnalyzeRequest(BaseModel):
    idea: str = Field(min_length=1)
    session_id: str | None = None
    answers: list[str] | None = None


class AnalyzeResponse(BaseModel):
    ready: bool
    requirements: Requirements | None = None
    questions: list[str] | None = None
    draft: Requirements | None = None
    session_id: str | None = None


class RDResearchRequest(BaseModel):
    requirements: Requirements
    session_id: str | None = None
    idea: str | None = None
    k: int = Field(default=4, ge=1, le=10, description="k per subagent")
    k_images: int = Field(default=1, ge=0, le=5)
