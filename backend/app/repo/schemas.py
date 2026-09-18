"""Schemas for codebase indexing (repo RAG)."""

from pydantic import BaseModel, Field


class RepoIndexRequest(BaseModel):
    path: str = Field(min_length=1, description="Local filesystem path of the repo root")
    repo_id: str | None = Field(default=None, description="Override the deterministic id")
    name: str | None = None


class RepoFile(BaseModel):
    path: str  # repo-relative posix path
    text: str


class CodeChunk(BaseModel):
    path: str
    language: str = "python"
    symbol: str = ""
    kind: str = "block"  # module | class | method | function | block
    start_line: int
    end_line: int
    imports: list[str] = Field(default_factory=list)
    text: str


class RepoSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=6, ge=1, le=20)
    repo_id: str | None = None
    path_prefix: str | None = None


class RepoStats(BaseModel):
    repo_id: str
    name: str
    root: str
    files: int
    chunks: int
    skipped: int
    runtime_ms: int


class RepoInfo(BaseModel):
    repo_id: str
    name: str
    root: str
    files: int
    chunks: int
    status: str
