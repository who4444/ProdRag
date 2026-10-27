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
    action: str = "create"  # create | modify (extend mode); demo mode ignores it


class SandboxResult(BaseModel):
    passed: bool
    pytest_log: str = ""
    demo_log: str = ""
    returncode: int = 0
    runtime_ms: int = 0  # whole sandbox (pip + pytest + demo)
    demo_runtime_ms: int = 0  # demo.py subprocess only
    memory_peak_kb: int | None = None  # ru_maxrss of the process tree (approximate, Linux)
    repo_log: str = ""  # extend mode: patch/copy log (repo files copied, changes applied)


class CodeArtifact(BaseModel):
    artifact_run_id: str = ""
    coding_spec: CodingSpec = Field(default_factory=CodingSpec)
    files: list[GeneratedFile] = Field(default_factory=list)
    test_file: GeneratedFile | None = None
    sandbox: SandboxResult | None = None
    created_at: float = 0


# --- Extend mode (PoC extension of an indexed/local codebase) ---

class ExtendFileTask(BaseModel):
    path: str
    # create = new file, modify = patch existing repo file
    action: str = "create"  # create | modify
    goal: str = ""
    # Full replacement content (modify: whole new file content) or
    # search/replace blocks for a minimal patch (both optional).
    content: str = ""
    search_replace: list[dict] = Field(default_factory=list)  # [{search, replace}]


class RepoContext(BaseModel):
    repo_id: str = ""
    root: str = ""
    mode: str = "repo"  # repo=indexed rag | local=direct path | none
    # repo-relative path -> file text (subset relevant to the task)
    files: dict[str, str] = Field(default_factory=dict)
    # retrieval hits used for grounding (path/symbol/lines/snippet)
    hits: list[dict] = Field(default_factory=list)


class ExtendSpec(BaseModel):
    entrypoint: str = ""
    focus_paths: list[str] = Field(default_factory=list)
    file_tasks: list[ExtendFileTask] = Field(default_factory=list)
    test_intents: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class ExtendArtifact(BaseModel):
    artifact_run_id: str = ""
    repo: RepoContext = Field(default_factory=RepoContext)
    extend_spec: ExtendSpec = Field(default_factory=ExtendSpec)
    changes: list[GeneratedFile] = Field(default_factory=list)  # action=create|modify
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


# Extend mode request/response
class CodeExtendRequest(BaseModel):
    idea: str = Field(min_length=1, description="What to add/change in the target codebase")
    repo_id: str | None = Field(default=None, description="Indexed repo (from /repo/index)")
    path: str | None = Field(default=None, description="OR a local filesystem path (copied into the sandbox)")
    goal: str = Field(default="run tests", description="Validation command intent; v1: pytest + optional entrypoint")
    entrypoint: str = Field(default="", description="Optional runnable module to exec after pytest (repo-relative)")
    session_id: str | None = None
