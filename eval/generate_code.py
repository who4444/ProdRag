"""Generate code-demo items by crossing demo intents with constraint sets.

Deterministic, offline. Each item carries requirements + acceptance_criteria
shaped like v2_golden code items so eval/run_live.py can synthesize a
ResearchArtifact directly.

Usage:
  uv run python eval/generate_code.py --n 10 --seed 42 --out eval/golden/v1/code.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval"))

from schemas import EvalItem  # noqa: E402

INTENTS: list[tuple[str, str, str, list[str], list[str]]] = [
    ("attention", "scaled dot-product attention", "demo attention core_algorithm", ["numpy"], ["demo.py exits 0", "output shape (seq, dim)", "pytest test_demo.py passes"]),
    ("chunking", "RecursiveCharacterTextSplitter 800/100", "demo chunking", [], ["demo.py exits 0", "chunk count >1", "overlap 100"]),
    ("bm25", "BM25 sparse retrieval", "demo BM25", ["numpy"], ["demo.py exits 0", "sparse index built", "query returns ranked"]),
    ("rrf", "reciprocal rank fusion of dense and sparse lists", "demo RRF fusion", ["numpy"], ["demo.py exits 0", "RRF fusion order correct", "prints JSON metrics"]),
    ("citation", "sentence-split citation coverage cited/total", "demo citation coverage", [], ["demo.py exits 0", "citation coverage computed", "invalid markers detected"]),
    ("gate", "deterministic verdict mapping from supported claims", "demo evidence gate", [], ["demo.py exits 0", "verdict mapping correct", "evidence gate enforced"]),
    ("metrics", "machine-checkable evaluation metrics subset assertion", "demo metrics contract", ["numpy"], ["demo.py exits 0", "metrics snake_case", "demo prints evaluation_metrics JSON"]),
    ("ledger", "ContextVar Usage ledger with price per MTok", "demo usage ledger", [], ["demo.py exits 0", "usage ledger counts calls", "cost_usd computed"]),
    ("sandbox", "subprocess sandbox with timeout", "demo sandbox", ["pytest"], ["demo.py exits 0", "test_demo.py exists", "sandbox passed true"]),
    ("ttl", "episodes with created_at filter", "demo TTL", ["numpy"], ["demo.py exits 0", "old episodes filtered", "recent returned"]),
]

CONSTRAINTS: list[list[str]] = [
    ["numpy only"],
    ["python only"],
    ["python only", "no GPU"],
    ["numpy only", "toy data"],
]


def generate(n: int, seed: int) -> list[EvalItem]:
    rng = random.Random(seed)
    combos = list(itertools.product(INTENTS, CONSTRAINTS))
    rng.shuffle(combos)
    now = time.time()
    items: list[EvalItem] = []
    for i, (intent, constraints) in enumerate(combos[:n]):
        slug, core_idea, goal, deps, criteria = intent
        constraints = list(constraints)
        items.append(
            EvalItem(
                id=f"code-gen-{seed}-{i + 1:02d}",
                type="code",
                idea=f"{slug} demo: {core_idea} with toy data, demo.py + test_demo.py",
                requirements={
                    "goal": goal,
                    "core_idea": core_idea,
                    "demo_type": "script",
                    "constraints": constraints,
                    "audience": "researcher",
                    "open_questions": [],
                },
                acceptance_criteria=criteria,
                expected_deps=deps,
                acceptance=[],
                baseline_eligible=True,
                generator="generate_code",
                generator_version="v1",
                created_at=now,
            )
        )
    # Keep stable order by id for frozen diffs.
    items.sort(key=lambda it: it.id)
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate code golden items")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    items = generate(args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.model_dump_json() + "\n")
    print(f"wrote {len(items)} items -> {out} (seed={args.seed})")


if __name__ == "__main__":
    main()
