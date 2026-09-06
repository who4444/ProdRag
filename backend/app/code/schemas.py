"""Code module schemas — CodingSpec, FileTask, GeneratedFile, SandboxResult, CodeArtifact."""

from pydantic import BaseModel, Field


class FileTask(BaseModel):
    path: str
    goal: str
    context_slice: str = ""


class CodingSpec(BaseModel):
    file_tasks: list[FileTask] = Field(default_factory=list)
    test_intents: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class GeneratedFile(BaseModel):
    path: str
    content: str


class SandboxResult(BaseModel):
    passed: bool
    pytest_log: str = ""
    demo_log: str = ""
    returncode: int = 0
    runtime_ms: int = 0


class CodeArtifact(BaseModel):
    artifact_run_id: str = ""
    coding_spec: CodingSpec = Field(default_factory=CodingSpec)
    files: list[GeneratedFile] = Field(default_factory=list)
    test_file: GeneratedFile | None = None
    sandbox: SandboxResult | None = None
    created_at: float = 0


# API schemas
class CodeAnalyzeRequest(BaseModel):
    artifact: dict = Field(description="ResearchArtifact dict")


class CodeAnalyzeResponse(BaseModel):
    coding_spec: CodingSpec
    file_tasks: list[FileTask] | None = None  # alias for convenience

    model_config = {"populate_by_name": True}


class CodeGenerateRequest(BaseModel):
    artifact: dict
    session_id: str | None = None
