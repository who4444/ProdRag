# Eval — automated dataset + live harness (v1)

Frozen golden set: `eval/golden/v1/` (55 items: 20 faith + 10 code + 20 feasibility + 5 runs) + `MANIFEST.json` (sha256). Never edit frozen files in place — bump to `v2/`.

## Quick start (live, real numbers)

```bash
# 1. deps + offline pre-flight
uv sync --extra dev --extra modal --extra frontend
PRODRAG_OFFLINE=1 uv run pytest -q

# 2. live stack (needs Supabase + DeepSeek + Modal embed in .env)
docker compose -f deploy/docker-compose.yml up --build
bash deploy/modal/manage.sh status   # prodrag-embed/clip warm (cold 30-60s)

# 3. index repos, capture repo_ids
curl -s -H "Authorization: Bearer $PRODRAG_API_TOKEN" -H "Content-Type: application/json" \
  -d '{"path":"/abs/ProdRag"}' localhost:8000/repo/index | jq
# ext repos (pinned SHA under eval/repos/): click + fastapi-sample

# 4. run live eval (~30-60 min full; --limit for smoke)
PRODRAG_API_TOKEN=... uv run python eval/run_live.py \
  --base-url http://localhost:8000 --golden eval/golden/v1 \
  --repo-map '{"self":"<id>","ext-a":"<id>","ext-b":"<id>"}' \
  --out eval/results/live-v1.json

# 5. baseline (k=1 minimal retrieval) + report
PRODRAG_API_TOKEN=... uv run python eval/run_live.py --baseline llm-only \
  --golden eval/golden/v1 --repo-map '{...}' --out eval/results/baseline-v1.json
uv run python eval/report.py eval/results/live-v1.json eval/results/baseline-v1.json \
  --out eval/results/SUMMARY.md
```

Smoke (10 items, ~5 min): add `--limit 3 --only faithfulness,code`.

## What is measured

| Layer | Quality | Perf | Cost |
|---|---|---|---|
| `faithfulness` via `POST /query` | `recall@k`, `MRR`, `citation_coverage` (≥50% sents `[N]`), `acceptance_hit` | `latency_ms` p50/p95 | answer chars, n_sources |
| `code` via `POST /code/generate` | `sandbox.passed` rate | `runtime_ms`, `demo_runtime_ms`, `memory_peak_kb` | 5 LLM calls/run |
| `feasibility` via `POST /feasibility/analyze` | `verdict_accuracy` (+lenient plausible↔with_risks), `citation_precision`, `unsupported_rate`, `must_cite_recall` | `latency_ms`, `n_events` | `llm_calls`, `cost_usd` from report usage |
| `run` via `POST /runs` | `status`, `verdict`, `worthiness`, `integration proposed|skipped` | `wall_ms`, `demo_runtime`, `memory` | `llm_calls`, `cost_usd` end-to-end |

Budgets: `--max-llm-calls 25 --max-wall-s 900` (server defaults in `runs/schemas.py`). `budget_exceeded` is a real outcome, not a harness failure. Interrupts auto-resume once (clarify canned answers, reframe accept).

## Regenerate / extend the dataset

```bash
uv run python eval/generate_faith.py --repo /abs/repo --repo-id self --n 15 --seed 42 --out /tmp/faith.jsonl
uv run python eval/generate_code.py --n 10 --seed 42 --out /tmp/code.jsonl
uv run python eval/generate_feasibility.py --repo /abs/repo --repo-id self --n 10 --seed 42 --out /tmp/feas.jsonl --validate
uv run python -c "import sys; sys.path.insert(0,'eval'); from schemas import write_manifest; print(write_manifest('eval/golden/v1').model_dump_json(indent=2))"
```

Generators are deterministic (`--seed`), offline, template-based. `generate_feasibility --validate` runs the real `citation.validate_ref + apply_citation_gate` so `must_cite_paths` resolve (infeasible ideas keep a resolvable path; the verdict expectation comes from the idea, the gate must still yield `unsupported` for the missing subsystem).

## Files

- `schemas.py` — `EvalItem`, `GoldenManifest`, `load_golden`, `write_manifest`
- `generate_{faith,code,feasibility}.py` — dataset miners (walker/chunker + templates)
- `run_live.py` — live runner (urllib NDJSON, no new deps)
- `report.py` — markdown table + workflow-vs-baseline delta
- `golden/v1/{faithfulness,code,feasibility,runs}.jsonl + MANIFEST.json` — frozen
- `results/*.json` — gitignored run outputs (see `.gitignore`)
- Legacy: `recall.py + golden.jsonl`, `v2_eval.py + v2_golden.jsonl` (kept; `run_live.py` supersedes)

## CI

`ruff check backend/app backend/tests eval` covers eval. `backend/tests/test_eval_spec.py` covers schemas/generators/runner helpers offline (no server). Live eval is nightly/on-demand, not per-PR (needs credentials + 30-60 min).
