"""Generate faithfulness items by mining repo symbols + docs.

Template-based by default (deterministic, offline). Optional --llm rewrites
questions via chat_json for retrieval-friendly phrasing (needs DeepSeek key).

Usage:
  uv run python eval/generate_faith.py --repo /abs/path --repo-id self --n 15 --seed 42 --out eval/golden/v1/faithfulness.jsonl
  uv run python eval/generate_faith.py --repo /abs/path --n 5 --seed 7 --out /tmp/faith.jsonl --no-validate
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

from app.repo.chunker import chunk_file  # noqa: E402
from app.repo.walker import walk_repo  # noqa: E402
from schemas import EvalItem  # noqa: E402

# Curated question templates per file family: (path_substring, question_tpl, acceptance, expected_answer_tpl).
TEMPLATES: list[tuple[str, str, list[str], str]] = [
    ("code/sandbox.py", "How does the code sandbox in {path} decide passed and what timeouts apply?", ["sandbox", "pytest", "demo"], "tmpdir+subprocess pip 60s/pytest+demo 15s; passed = demo 0 and pytest in (0,127)"),
    ("rag/retrieval.py", "Explain hybrid retrieval in {path} (dense+BM25+RRF) and the rerank fallback", ["dense", "BM25", "RRF"], "dense bge-m3 1024 + BM25 sparse fused via RRF hybrid_k=12, cross-encoder rerank optional"),
    ("feasibility/citation.py", "What does the citation gate in {path} require for a claim to be supported?", ["valid", "path"], "path must resolve in repo with valid line range; provisional -> unsupported; technical/integration need .py ref"),
    ("feasibility/judge.py", "How does {path} map dimension plausibility to a final verdict?", ["plausible", "infeasible"], "0 implausible -> plausible, 1 -> plausible_with_risks, 2+ -> infeasible"),
    ("core/llm.py", "How does {path} track LLM usage and cost per run?", ["usage", "cost"], "Usage{calls,prompt,completion,cost} via record_usage; per-run ledgers via activate_usage"),
    ("repo/indexer.py", "How does repo indexing in {path} work and what is RepoStats?", ["chunks", "files"], "walk -> chunk -> embed batched -> code_chunks; RepoStats{files,chunks,skipped,runtime_ms}"),
    ("rd/subagent.py", "What retrieval fallbacks does the R&D subagent in {path} apply?", ["web_search", "HyDE"], "rag_search then web_search when <2 hits, HyDE rewrite when 0, confidence high>=3/med>=1/low"),
    ("rd/validator.py", "What does the validator in {path} emit and does it block?", ["coverage", "risks"], "Validation{coverage,missing,feasible,risks}; emit-with-risks never blocks"),
    ("code/tester.py", "Where does test_demo.py come from according to {path}?", ["acceptance_criteria", "test_intents"], "from CodingSpec.test_intents + demo_spec.acceptance_criteria, not from demo.py"),
    ("runs/evaluate.py", "How does {path} decide verdict and worthiness?", ["pass", "worth"], "sandbox fail -> fail; hypotheses unsupported -> pass_with_caveats; worthiness worth_it|marginal|not_worth_it"),
    ("runs/graph.py", "What is the node order and budget handling in {path}?", ["budget", "finalize"], "understand->map->assess->judge->research->poc->evaluate->integration->finalize; budget edges to finalize"),
    ("core/vectorstore.py", "Which Qdrant collections exist and how is hybrid search done in {path}?", ["text_chunks", "RRF"], "text_chunks/image_chunks/episodes/code_chunks; dense+BM25 prefetch fused via RRF"),
]


def pick_chunks(root: Path, seed: int, n: int) -> list[dict]:
    files, _ = walk_repo(root)
    by_path = {f.path: f for f in files}
    # Prefer files matching template families for stable, meaningful items.
    scored: list[tuple[int, str]] = []
    for path in by_path:
        bonus = next((i for i, (sub, _, _, _) in enumerate(TEMPLATES) if sub in path), 999)
        scored.append((bonus, path))
    scored.sort()
    rng = random.Random(seed)
    # Take top pool then sample deterministically.
    pool = [p for _, p in scored[: max(n * 3, 12)]]
    rng.shuffle(pool)
    chunks: list[dict] = []
    fallback: list[dict] = []
    for path in pool:
        for c in chunk_file(by_path[path]):
            entry = {"path": path, "symbol": c.symbol or path, "kind": c.kind, "start": c.start_line, "end": c.end_line}
            if len(c.text.strip()) < 40:
                continue
            if c.kind in ("function", "method", "class") and c.symbol:
                chunks.append(entry)
            else:
                fallback.append(entry)
            if len(chunks) >= n * 2:
                break
        if len(chunks) >= n * 2:
            break
    if len(chunks) < n and fallback:
        rng.shuffle(fallback)
        chunks += fallback[: n - len(chunks)]
    rng.shuffle(chunks)
    return chunks[:n]


def template_for(path: str) -> tuple[str, list[str], str]:
    for sub, tpl, acc, ans in TEMPLATES:
        if sub in path:
            return tpl, acc, ans
    return (
        "What does {symbol} in {path} implement ({kind} lines {start}-{end})?",
        [path.split("/")[-1].replace(".py", "")],
        "see source chunk for the implementation",
    )


def generate(repo: str, repo_id: str, n: int, seed: int) -> list[EvalItem]:
    root = Path(repo).expanduser().resolve()
    chunks = pick_chunks(root, seed, n)
    items: list[EvalItem] = []
    now = time.time()
    for i, c in enumerate(chunks):
        tpl, acc, ans = template_for(c["path"])
        question = tpl.format(path=c["path"], symbol=c["symbol"], kind=c["kind"], start=c["start"], end=c["end"])
        items.append(
            EvalItem(
                id=f"faith-gen-{seed}-{i + 1:02d}",
                type="faithfulness",
                repo_id=repo_id,
                repo_root=str(root),
                question=question,
                expected_sources=[c["path"]],
                must_cite_paths=[],
                expected_answer=ans,
                acceptance=acc,
                baseline_eligible=True,
                generator="generate_faith",
                generator_version="v1",
                created_at=now,
            )
        )
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate faithfulness golden items from a repo")
    ap.add_argument("--repo", required=True, help="Local repo root to mine")
    ap.add_argument("--repo-id", default="self")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    items = generate(args.repo, args.repo_id, args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.model_dump_json() + "\n")
    print(f"wrote {len(items)} items -> {out} (seed={args.seed} repo={args.repo})")


if __name__ == "__main__":
    main()
