"""Eval dataset schemas — frozen golden-set items + run results.

Versioned under eval/golden/v1/*.jsonl. Never overwrite a frozen version in
place; bump to v2/ and record the manifest sha256.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

EvalType = Literal["faithfulness", "code", "feasibility", "run"]
Verdict = Literal["plausible", "plausible_with_risks", "infeasible"]


class EvalItem(BaseModel):
    id: str
    type: EvalType
    repo_id: str | None = Field(default=None, description="Deterministic repo id or alias (self/ext-a/ext-b)")
    repo_root: str | None = Field(default=None, description="Resolved root at generation time (informational)")
    question: str | None = None
    idea: str | None = None
    requirements: dict | None = None
    expected_sources: list[str] = Field(default_factory=list)
    must_cite_paths: list[str] = Field(default_factory=list)
    expected_verdict: Verdict | None = None
    expected_answer: str | None = None
    acceptance: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    expected_deps: list[str] = Field(default_factory=list)
    dimension: str | None = Field(default=None, description="Primary feasibility dimension, if any")
    baseline_eligible: bool = True
    generator: str = "hand"
    generator_version: str = "v1"
    created_at: float = 0.0


class GoldenManifest(BaseModel):
    version: str = "v1"
    files: dict[str, str] = Field(default_factory=dict, description="filename -> sha256")
    counts: dict[str, int] = Field(default_factory=dict)
    generator_version: str = "v1"


def item_sha(item: dict) -> str:
    return hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()[:12]


def load_golden(golden_dir: str | Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    golden_dir = Path(golden_dir)
    for path in sorted(golden_dir.glob("*.jsonl")):
        if path.name.startswith("MANIFEST"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(EvalItem.model_validate(json.loads(line)))
    return items


def write_manifest(golden_dir: str | Path, version: str = "v1") -> GoldenManifest:
    golden_dir = Path(golden_dir)
    files: dict[str, str] = {}
    counts: dict[str, int] = {}
    for path in sorted(golden_dir.glob("*.jsonl")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[path.name] = digest
        counts[path.name] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    manifest = GoldenManifest(version=version, files=files, counts=counts)
    (golden_dir / "MANIFEST.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
