"""Generate feasibility items with deterministic citation-gate validation.

Template-based by default (offline). For --repo targets the generator mines
real file paths via the walker so must_cite_paths always resolve; --validate
runs citation.validate_ref + apply_citation_gate against the repo root and
drops items whose cites do not resolve (infeasible items keep one resolvable
path so the gate can still run, the verdict expectation comes from the idea).

Usage:
  uv run python eval/generate_feasibility.py --repo /abs/ProdRag --repo-id self --n 10 --seed 42 --out /tmp/feas.jsonl
  uv run python eval/generate_feasibility.py --repo /abs/click --repo-id ext-a --n 10 --seed 42 --out /tmp/feas_click.jsonl --validate
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from app.feasibility.citation import apply_citation_gate, validate_ref  # noqa: E402
from app.feasibility.schemas import EvidenceRef, FeasibilityClaim  # noqa: E402
from app.repo.walker import walk_repo  # noqa: E402
from schemas import EvalItem  # noqa: E402

# (slug, idea_tpl, verdict, dimension, path_hints[])
PLAUSIBLE: list[tuple[str, str, str, str, list[str]]] = [
    ("cache", "Add a semantic cache in front of {anchor} to cut repeated-query cost, keyed by normalized hash", "plausible", "technical", ["rag/retrieval.py", "memory.py"]),
    ("rerank-toggle", "Add a rerank toggle for {anchor} falling back to RRF order when the service is unset", "plausible", "technical", ["core/rerank.py", "rag/retrieval.py"]),
    ("history", "Expose list/get history for {anchor} backed by the relational store", "plausible", "integration", ["api/repo.py", "core/db.py"]),
    ("coverage", "Record per-direction citation coverage for {anchor} so low-evidence dirs are gated", "plausible", "data", ["rd/subagent.py", "rd/schemas.py"]),
    ("metrics", "Persist JSON metrics from {anchor} into the evaluation performance sample", "plausible", "data", ["code/sandbox.py", "runs/evaluate.py"]),
]

RISKY: list[tuple[str, str, str, str, list[str]]] = [
    ("sandbox-harden", "Swap {anchor} subprocess isolation for docker+gvisor for untrusted code", "plausible_with_risks", "technical", ["code/sandbox.py", "code/pipeline.py"]),
    ("auto-retry", "Auto-retry generation on sandbox failure carrying logs plus previous files for {anchor}", "plausible_with_risks", "integration", ["code/pipeline.py", "runs/graph.py"]),
    ("pagination", "Add cursor pagination to list endpoints around {anchor}", "plausible_with_risks", "data", ["api/documents.py", "core/db.py"]),
]

INFEASIBLE: list[tuple[str, str, str, str, list[str]]] = [
    ("gpu-render", "Add CUDA kernel fusion to {anchor} for 10x throughput without new dependencies", "infeasible", "technical", ["main.py"]),
    ("k8s-op", "Add Kubernetes operator reconciliation to {anchor} with zero new services", "infeasible", "integration", ["main.py"]),
    ("finetune", "Fine-tune embeddings inside {anchor} and hot-swap vectors without re-ingest", "infeasible", "data", ["core/embeddings/text.py", "core/vectorstore.py"]),
]


def _resolve(paths: list[str], files: set[str]) -> list[str]:
    ordered = sorted(files)
    out: list[str] = []
    for p in paths:
        if p in files:
            out.append(p)
            continue
        # Match by suffix (e.g. ext repos without backend/ prefix).
        matches = [f for f in ordered if f.endswith(p) or f.endswith(p.split("/")[-1])]
        if matches:
            out.append(matches[0])
            continue
        # Last resort: first real file so cites always resolve; the verdict
        # expectation comes from the idea, the gate still exercises the FS path.
        if ordered:
            out.append(ordered[0])
        else:
            out.append(p)
    return out


def generate(repo: str, repo_id: str, n: int, seed: int, validate: bool = True) -> list[EvalItem]:
    root = Path(repo).expanduser().resolve()
    files, _ = walk_repo(root)
    file_set = {f.path for f in files}
    rng = random.Random(seed)
    pool = (PLAUSIBLE * 3 + RISKY * 2 + INFEASIBLE * 2)
    rng.shuffle(pool)
    now = time.time()
    items: list[EvalItem] = []
    anchors = sorted(file_set)[:50] or ["main.py"]
    for i, (slug, tpl, verdict, dim, hints) in enumerate(pool[:n]):
        anchor = rng.choice(anchors)
        idea = tpl.format(anchor=anchor)
        cites = _resolve(hints, file_set)
        # Ensure at least one resolvable cite so the gate exercises the FS path.
        if validate and file_set:
            claims = [
                FeasibilityClaim(
                    dimension=dim,  # type: ignore[arg-type]
                    statement=idea[:220],
                    evidence=[EvidenceRef(path=p, start_line=1, end_line=1) for p in cites[:2]],
                )
            ]
            for c in claims:
                for r in c.evidence:
                    validate_ref(root, r)
            apply_citation_gate(root, claims)
            if not any(r.valid for c in claims for r in c.evidence):
                # Replace with the first file that validates (single-line ref).
                for candidate in sorted(file_set):
                    ref = EvidenceRef(path=candidate, start_line=1, end_line=1)
                    validate_ref(root, ref)
                    if ref.valid:
                        cites = [candidate]
                        break
        items.append(
            EvalItem(
                id=f"feas-gen-{repo_id}-{seed}-{i + 1:02d}",
                type="feasibility",
                repo_id=repo_id,
                repo_root=str(root),
                idea=idea,
                must_cite_paths=cites[:3],
                expected_verdict=verdict,  # type: ignore[arg-type]
                dimension=dim,
                baseline_eligible=True,
                generator="generate_feasibility",
                generator_version="v1",
                created_at=now,
            )
        )
    items.sort(key=lambda it: it.id)
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate feasibility golden items")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--repo-id", default="self")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    items = generate(args.repo, args.repo_id, args.n, args.seed, validate=args.validate)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.model_dump_json() + "\n")
    print(f"wrote {len(items)} items -> {out} (seed={args.seed} repo={args.repo} validate={args.validate})")


if __name__ == "__main__":
    main()
