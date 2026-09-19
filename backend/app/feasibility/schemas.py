"""Feasibility schemas — idea understanding, codebase map, claims, report."""

from typing import Literal

from pydantic import BaseModel, Field

Dimension = Literal["technical", "integration", "data"]
DIMENSION_IDS: tuple[str, ...] = ("technical", "integration", "data")


class EvidenceRef(BaseModel):
    path: str = Field(description="Repo-relative path, e.g. backend/app/rd/subagent.py")
    start_line: int | None = None
    end_line: int | None = None
    snippet: str = ""
    valid: bool | None = Field(default=None, description="Set by the citation gate, not the LLM")
    reason: str = Field(default="", description="Why invalid, when valid is False")


class IdeaUnderstanding(BaseModel):
    interpretation: str = ""
    problem: str = ""
    expected_outcome: str = ""
    assumptions: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)


class FeasibilityClaim(BaseModel):
    id: str = ""
    dimension: Dimension = "technical"
    statement: str = ""
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"
    status: Literal["supported", "assumption", "unsupported"] = "assumption"
    provisional: bool = Field(
        default=False,
        description="Set by failure fallbacks; the gate always treats these as unsupported",
    )


class DimensionVerdict(BaseModel):
    plausible: bool = False
    rationale: str = ""
    blockers: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


class ExtensionPoint(BaseModel):
    path: str
    symbol: str = ""
    reason: str = ""
    evidence: list[EvidenceRef] = Field(default_factory=list)


class CodebaseMap(BaseModel):
    modules: list[dict] = Field(default_factory=list)  # {path, symbols, loc}
    entrypoints: list[str] = Field(default_factory=list)
    dependency_edges: list[list[str]] = Field(default_factory=list)  # [from_path, to_path]
    impacted_modules: list[str] = Field(default_factory=list)
    extension_points: list[ExtensionPoint] = Field(default_factory=list)
    data_flow: str = ""
    data_available: list[str] = Field(default_factory=list)
    data_missing: list[str] = Field(default_factory=list)
    data_cleanliness: str = "unknown"


class FeasibilityReport(BaseModel):
    run_id: str = ""
    repo_id: str = ""
    idea: str = ""
    understanding: IdeaUnderstanding = Field(default_factory=IdeaUnderstanding)
    codebase_map: CodebaseMap = Field(default_factory=CodebaseMap)
    claims: list[FeasibilityClaim] = Field(default_factory=list)
    dimensions: dict[str, DimensionVerdict] = Field(default_factory=dict)
    verdict: Literal["plausible", "plausible_with_risks", "infeasible"] = "infeasible"
    blockers: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    reframed_idea: str | None = None
    usage: dict = Field(default_factory=dict)
    created_at: float = 0


# API request/response schemas


class UnderstandRequest(BaseModel):
    idea: str = Field(min_length=1)
    answers: list[str] | None = None
    session_id: str | None = None


class UnderstandResponse(BaseModel):
    ready: bool
    understanding: IdeaUnderstanding | None = None
    questions: list[str] | None = None


class AnalyzeRequest(BaseModel):
    idea: str = Field(min_length=1)
    repo_id: str = Field(min_length=1)
    understanding: IdeaUnderstanding | None = None
    answers: list[str] | None = None
    k: int = Field(default=6, ge=1, le=20)
