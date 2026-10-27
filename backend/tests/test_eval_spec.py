"""Eval automation: schemas, generators, runner helpers (offline, no server)."""

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
GOLDEN_V1 = EVAL_DIR / "golden" / "v1"
sys.path.insert(0, str(EVAL_DIR))

from schemas import EvalItem, load_golden, write_manifest  # noqa: E402


def test_golden_v1_counts_and_ids():
    items = load_golden(GOLDEN_V1)
    by_type: dict[str, list] = {}
    for it in items:
        by_type.setdefault(it.type, []).append(it)
    assert len(by_type["faithfulness"]) == 20
    assert len(by_type["code"]) == 10
    assert len(by_type["feasibility"]) == 20
    assert len(by_type["run"]) == 5
    ids = [it.id for it in items]
    assert len(ids) == len(set(ids)), "ids must be unique"


def test_golden_items_validate_and_have_expectations():
    for it in load_golden(GOLDEN_V1):
        assert isinstance(it, EvalItem)
        if it.type == "faithfulness":
            assert it.question and it.expected_sources
        elif it.type == "code":
            assert it.idea and it.acceptance_criteria
        elif it.type == "feasibility":
            assert it.idea and it.expected_verdict in ("plausible", "plausible_with_risks", "infeasible")
            assert it.must_cite_paths
        elif it.type == "run":
            assert it.idea and it.repo_id


def test_manifest_matches_files():
    manifest = json.loads((GOLDEN_V1 / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "v1"
    assert set(manifest["files"]) == {"faithfulness.jsonl", "code.jsonl", "feasibility.jsonl", "runs.jsonl"}
    assert manifest["counts"] == {"code.jsonl": 10, "faithfulness.jsonl": 20, "feasibility.jsonl": 20, "runs.jsonl": 5}
    import hashlib

    for name, digest in manifest["files"].items():
        assert hashlib.sha256((GOLDEN_V1 / name).read_bytes()).hexdigest() == digest


def test_write_manifest_roundtrip(tmp_path):
    import shutil

    dst = tmp_path / "golden"
    shutil.copytree(GOLDEN_V1, dst, ignore=shutil.ignore_patterns("MANIFEST.json"))
    m = write_manifest(dst)
    assert m.counts["faithfulness.jsonl"] == 20
    assert (dst / "MANIFEST.json").exists()


def test_generate_code_deterministic():
    sys.path.insert(0, str(EVAL_DIR))
    import generate_code

    a = generate_code.generate(10, 42)
    b = generate_code.generate(10, 42)
    assert [it.model_dump(exclude={"created_at"}) for it in a] == [it.model_dump(exclude={"created_at"}) for it in b]
    assert len(a) == 10
    assert all(it.type == "code" and it.acceptance_criteria for it in a)


def test_generate_faith_mines_real_repo(tmp_path):
    import generate_faith

    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "sandbox.py").write_text("def run_sandbox(files, timeout=15):\n    return True\n" * 10)
    (repo / "pkg" / "retrieval.py").write_text("def search_kb(question, k=4):\n    return [], []\n" * 10)
    items = generate_faith.generate(str(repo), "self", 2, 7)
    assert len(items) == 2
    assert all(it.type == "faithfulness" and it.question and it.expected_sources for it in items)
    # Deterministic across runs.
    again = generate_faith.generate(str(repo), "self", 2, 7)
    assert [i.id for i in items] == [i.id for i in again]


def test_generate_feasibility_cites_resolve(tmp_path):
    import generate_feasibility
    from app.feasibility.citation import validate_ref
    from app.feasibility.schemas import EvidenceRef

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("x = 1\n" * 20)
    (repo / "rag").mkdir()
    (repo / "rag" / "retrieval.py").write_text("y = 2\n" * 20)
    items = generate_feasibility.generate(str(repo), "self", 4, 42, validate=True)
    assert len(items) == 4
    for it in items:
        assert it.must_cite_paths
        for p in it.must_cite_paths:
            ref = EvidenceRef(path=p, start_line=1, end_line=1)
            validate_ref(repo, ref)
            assert ref.valid, f"cite must resolve: {p}"


def test_run_live_helpers_offline():
    sys.path.insert(0, str(EVAL_DIR))
    import run_live

    cov, cited, total = run_live.citation_stats("First claim [1]. Second claim [2]. Bare claim.", 2)
    assert (cited, total) == (2, 3)
    assert cov == round(2 / 3, 3)
    assert run_live.citation_stats("", 0) == (0.0, 0, 0)

    item = EvalItem(id="code-x", type="code", idea="demo", requirements={"goal": "g", "core_idea": "c", "demo_type": "script", "constraints": [], "audience": "researcher", "open_questions": []}, acceptance_criteria=["demo.py exits 0"], expected_deps=["numpy"])
    art = run_live.artifact_for(item)
    assert art["demo_spec"]["acceptance_criteria"] == ["demo.py exits 0"]
    assert "numpy" in art["demo_spec"]["tech_stack"]

    agg = run_live.aggregate(
        [
            {"id": "a", "type": "faithfulness", "recall_hit": True, "mrr": 1.0, "citation_pass": True, "acceptance_hit": True, "latency_ms": 100},
            {"id": "b", "type": "faithfulness", "recall_hit": False, "mrr": 0.0, "citation_pass": False, "acceptance_hit": False, "latency_ms": 200},
            {"id": "c", "type": "code", "sandbox_passed": True, "latency_ms": 500},
            {"id": "d", "type": "feasibility", "verdict_match": True, "verdict_match_lenient": True, "n_claims": 2, "n_unsupported": 0, "latency_ms": 300},
            {"id": "e", "type": "run", "status": "done", "sandbox_passed": True, "cost_usd": 0.01, "llm_calls": 5, "latency_ms": 900},
        ]
    )
    assert agg["faithfulness"]["recall"] == 0.5
    assert agg["code"]["sandbox_pass_rate"] == 1.0
    assert agg["feasibility"]["verdict_accuracy"] == 1.0
    assert agg["run"]["done_rate"] == 1.0
    assert agg["errors"] == 0


def test_report_renders_table(tmp_path):
    sys.path.insert(0, str(EVAL_DIR))
    import report
    import run_live

    results = [{"id": "a", "type": "faithfulness", "recall_hit": True, "mrr": 1.0, "citation_pass": True, "acceptance_hit": True, "latency_ms": 100}]
    payload = {"config": {"base_url": "http://x", "golden": "g", "baseline": "off", "k": 5, "n_items": 1, "health": {}, "search": {}}, "aggregate": run_live.aggregate(results), "items": results, "elapsed_s": 1.0}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    md, _ = report.render(str(p))
    assert "| RAG /query |" in md and "| Code /code/generate |" in md
