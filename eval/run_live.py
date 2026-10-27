"""Live eval runner — exercises the running stack end-to-end and records real numbers.

Covers: /query (recall/MRR/citation/acceptance + latency), /code/generate
(sandbox pass + runtime/memory), /feasibility/analyze (verdict accuracy +
citation precision), /runs (status/verdict/worthiness/cost/wall_ms).

Baseline mode (--baseline llm-only) re-runs with k=1 minimal retrieval and
marks results baseline=true, approximating an LLM-only comparison (the server
has no retrieval-off flag; true LLM-only would need one).

Usage:
  PRODRAG_API_TOKEN=... uv run python eval/run_live.py --base-url http://localhost:8000 \\
      --golden eval/golden/v1 --out eval/results/live-v1.json
  PRODRAG_API_TOKEN=... uv run python eval/run_live.py --base-url http://localhost:8000 \\
      --golden eval/golden/v1 --baseline llm-only --out eval/results/baseline-v1.json --limit 10
  uv run python eval/run_live.py --help

Repo aliases (self/ext-a/ext-b) resolve via --repo-map '{"self":"<id>"}'.
When unmapped the alias is sent as-is; the server 400 is recorded as an error
(real number, not a crash). Index repos first via POST /repo/index.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval"))

from schemas import load_golden  # noqa: E402

SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
CITE_RE = re.compile(r"\[(\d+)\]")


def _headers(token: str) -> dict:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}


def get_json(base: str, path: str, token: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(base + path, headers={"Authorization": f"Bearer {token}"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"ok": True, "status": resp.status, "body": json.loads(resp.read().decode() or "{}")}
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode() or "{}")
        except Exception:
            body = {}
        return {"ok": False, "status": e.code, "body": body}
    except Exception as e:
        return {"ok": False, "status": 0, "body": {"error": str(e)}}


def post_json(base: str, path: str, payload: dict, token: str, timeout: int) -> tuple(dict | None, float, str):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers=_headers(token), method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
        return json.loads(raw or "{}"), (time.perf_counter() - t0) * 1000, ""
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode()[:2000]
        except Exception:
            body = ""
        return None, (time.perf_counter() - t0) * 1000, f"http-{e.code}: {body[:300]}"
    except Exception as e:
        return None, (time.perf_counter() - t0) * 1000, f"{type(e).__name__}: {e}"[:500]


def post_ndjson(base: str, path: str, payload: dict, token: str, timeout: int) -> tuple[list[dict], float, str]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers=_headers(token), method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
        events = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
        return events, (time.perf_counter() - t0) * 1000, ""
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode()[:2000]
        except Exception:
            body = ""
        return [], (time.perf_counter() - t0) * 1000, f"http-{e.code}: {body[:300]}"
    except Exception as e:
        return [], (time.perf_counter() - t0) * 1000, f"{type(e).__name__}: {e}"[:500]


def citation_stats(text: str, n_sources: int) -> tuple[float, int, int]:
    sents = [s.strip() for s in SENT_RE.split((text or "").strip()) if s.strip()]
    if not sents or n_sources == 0:
        return 0.0, 0, len(sents)
    cited = sum(1 for s in sents if any(1 <= int(n) <= n_sources for n in CITE_RE.findall(s)))
    return round(cited / len(sents), 3), cited, len(sents)


def eval_faith(base: str, item, token: str, k: int, timeout: int) -> dict:
    payload = {"question": item.question, "k": k}
    events, latency_ms, err = post_ndjson(base, "/query", payload, token, timeout)
    if err:
        return {"id": item.id, "type": "faithfulness", "error": err, "latency_ms": round(latency_ms)}
    sources = []
    text = ""
    if events:
        first = events[0]
        if isinstance(first, dict):
            sources = first.get("items", []) or first.get("data", {}).get("items", []) or []
        for evt in events[1:]:
            if isinstance(evt, dict):
                text += evt.get("delta", "") or evt.get("text", "") or ""
    source_names = [s.get("source", "") for s in sources if isinstance(s, dict)]
    recall = any(any(exp in src for src in source_names) for exp in (item.expected_sources or [])) if item.expected_sources else None
    rank = None
    for i, src in enumerate(source_names):
        if any(exp in src for exp in (item.expected_sources or [])):
            rank = i + 1
            break
    mrr = round(1.0 / rank, 3) if rank else 0.0
    cov, cited, total = citation_stats(text, len(sources))
    acc = all(a.lower() in text.lower() for a in (item.acceptance or [])) if item.acceptance else None
    return {
        "id": item.id,
        "type": "faithfulness",
        "recall_hit": recall,
        "mrr": mrr,
        "citation_coverage": cov,
        "cited_sents": cited,
        "total_sents": total,
        "citation_pass": cov >= 0.5 if total else None,
        "acceptance_hit": acc,
        "n_sources": len(sources),
        "answer_chars": len(text),
        "latency_ms": round(latency_ms),
    }


def artifact_for(item) -> dict:
    req = item.requirements or {"goal": item.idea or "demo", "core_idea": item.idea or "demo", "demo_type": "script", "constraints": [], "audience": "researcher", "open_questions": []}
    idea = item.idea or req.get("goal", "demo")
    criteria = item.acceptance_criteria or ["demo.py exits 0"]
    return {
        "requirements": req,
        "summaries": [
            {"direction_id": "core_algorithm", "findings": req.get("core_idea", idea), "sources": [], "confidence": "high", "gaps": []},
            {"direction_id": "evaluation", "findings": "; ".join(criteria), "sources": [], "confidence": "high", "gaps": []},
        ],
        "validation": {"coverage": {"core_algorithm": True}, "missing": [], "feasible": True, "risks": []},
        "demo_spec": {
            "entrypoint": "demo.py",
            "tech_stack": ["python"] + list(item.expected_deps or []),
            "core_algorithm": req.get("core_idea", idea),
            "pseudocode": "",
            "acceptance_criteria": criteria,
        },
        "sources": [],
        "created_at": time.time(),
    }


def eval_code(base: str, item, token: str, timeout: int) -> dict:
    events, latency_ms, err = post_ndjson(base, "/code/generate", {"artifact": artifact_for(item)}, token, timeout)
    if err:
        return {"id": item.id, "type": "code", "error": err, "latency_ms": round(latency_ms)}
    if not events:
        return {"id": item.id, "type": "code", "error": "empty stream", "latency_ms": round(latency_ms)}
    last = events[-1]
    data = (last.get("data") or last.get("code_artifact") or last) if isinstance(last, dict) else {}
    sandbox = data.get("sandbox") or {}
    if not sandbox and isinstance(data.get("code_artifact"), dict):
        sandbox = data["code_artifact"].get("sandbox", {})
    files = data.get("files", []) or []
    return {
        "id": item.id,
        "type": "code",
        "sandbox_passed": bool(sandbox.get("passed")),
        "returncode": sandbox.get("returncode"),
        "runtime_ms": sandbox.get("runtime_ms"),
        "demo_runtime_ms": sandbox.get("demo_runtime_ms"),
        "memory_peak_kb": sandbox.get("memory_peak_kb"),
        "n_files": len(files),
        "pytest_tail": str(sandbox.get("pytest_log", ""))[-300:],
        "latency_ms": round(latency_ms),
    }


def eval_feasibility(base: str, item, token: str, repo_map: dict, k: int, timeout: int) -> dict:
    repo_id = repo_map.get(item.repo_id or "", item.repo_id or "")
    payload = {"idea": item.idea, "repo_id": repo_id, "k": k}
    events, latency_ms, err = post_ndjson(base, "/feasibility/analyze", payload, token, timeout)
    if err:
        return {"id": item.id, "type": "feasibility", "error": err, "latency_ms": round(latency_ms), "repo_id": repo_id}
    clarified = False
    if any(isinstance(e, dict) and e.get("event") == "clarify" for e in events) and not any(
        isinstance(e, dict) and (e.get("event") == "report" or "verdict" in e) for e in events
    ):
        # Complete the handshake like a real client: re-POST with canned answers.
        payload2 = {"idea": item.idea, "repo_id": repo_id, "k": k, "answers": ["the retrieval layer", "NDCG@10 with 5% lift", "offline batch tuning only"]}
        events2, lat2, err2 = post_ndjson(base, "/feasibility/analyze", payload2, token, timeout)
        latency_ms += lat2
        clarified = True
        if err2:
            return {"id": item.id, "type": "feasibility", "error": f"clarify-retry: {err2}", "latency_ms": round(latency_ms), "repo_id": repo_id, "clarified": clarified}
        events = events2
    report = None
    for evt in reversed(events):
        if isinstance(evt, dict) and (evt.get("event") == "report" or "verdict" in evt or "report" in evt):
            report = evt.get("report") or evt.get("data") or evt
            if isinstance(report, dict) and "verdict" not in report and isinstance(evt.get("report"), dict):
                report = evt["report"]
            break
    if report is None:
        # Fallback: last event may carry the report inline.
        report = events[-1] if events else {}
        if isinstance(report, dict) and "report" in report and isinstance(report["report"], dict):
            report = report["report"]
    verdict = report.get("verdict") if isinstance(report, dict) else None
    claims = report.get("claims", []) if isinstance(report, dict) else []
    n_supported = sum(1 for c in claims if isinstance(c, dict) and c.get("status") == "supported")
    n_unsupported = sum(1 for c in claims if isinstance(c, dict) and c.get("status") == "unsupported")
    n_assumption = sum(1 for c in claims if isinstance(c, dict) and c.get("status") == "assumption")
    n_provisional = sum(1 for c in claims if isinstance(c, dict) and c.get("provisional"))
    total_refs = sum(len(c.get("evidence", []) or []) for c in claims if isinstance(c, dict))
    valid_refs = sum(1 for c in claims if isinstance(c, dict) for r in (c.get("evidence", []) or []) if isinstance(r, dict) and r.get("valid"))
    usage = report.get("usage", {}) if isinstance(report, dict) else {}
    exp = item.expected_verdict
    match = (verdict == exp) if exp else None
    # Lenient match: plausible_with_risks counts for plausible and vice versa.
    lenient = None
    if exp and verdict:
        lenient = (verdict == exp) or ({verdict, exp} <= {"plausible", "plausible_with_risks"})
    cites_hit = None
    if item.must_cite_paths:
        blob = json.dumps(report)
        cites_hit = sum(1 for p in item.must_cite_paths if p in blob) / len(item.must_cite_paths)
        cites_hit = round(cites_hit, 3)
    return {
        "id": item.id,
        "type": "feasibility",
        "repo_id": repo_id,
        "clarified": clarified,
        "verdict": verdict,
        "expected_verdict": exp,
        "verdict_match": match,
        "verdict_match_lenient": lenient,
        "n_claims": len(claims),
        "n_supported": n_supported,
        "n_unsupported": n_unsupported,
        "n_assumption": n_assumption,
        "n_provisional": n_provisional,
        "citation_precision": round(valid_refs / total_refs, 3) if total_refs else None,
        "must_cite_recall": cites_hit,
        "llm_calls": usage.get("calls"),
        "cost_usd": usage.get("cost_usd"),
        "n_events": len(events),
        "latency_ms": round(latency_ms),
    }


def eval_run(base: str, item, token: str, repo_map: dict, k: int, max_calls: int, max_wall: int, timeout: int) -> dict:
    repo_id = repo_map.get(item.repo_id or "", item.repo_id or "")
    payload = {"idea": item.idea, "repo_id": repo_id, "k": k, "max_llm_calls": max_calls, "max_wall_s": max_wall}
    events, latency_ms, err = post_ndjson(base, "/runs", payload, token, timeout)
    if err:
        return {"id": item.id, "type": "run", "error": err, "latency_ms": round(latency_ms), "repo_id": repo_id}
    interrupts = [e for e in events if isinstance(e, dict) and e.get("event") == "interrupt"]
    run_id = None
    for e in events:
        if isinstance(e, dict) and e.get("run_id"):
            run_id = e["run_id"]
            break
    result = None
    for e in reversed(events):
        if isinstance(e, dict) and e.get("event") == "result":
            result = e.get("data") or e.get("result") or e
            break
    # Auto-resume up to 3 times (clarify -> reframe chains) so runs complete unattended.
    resumes = 0
    while resumes < 3:
        interrupts = [e for e in events if isinstance(e, dict) and e.get("event") == "interrupt"]
        run_id = next((e["run_id"] for e in events if isinstance(e, dict) and e.get("run_id")), None)
        result = next(
            (
                (e.get("data") or e.get("result") or e)
                for e in reversed(events)
                if isinstance(e, dict) and e.get("event") == "result"
            ),
            None,
        )
        if result is not None or not interrupts or not run_id:
            break
        pending = [e for e in events if isinstance(e, dict) and e.get("event") == "interrupt"]
        last = pending[-1]
        rpath = f"/runs/{run_id}/resume"
        if "reframe" in json.dumps(last):
            payload2: dict = {"accept": True}
        else:
            payload2 = {"answers": ["the retrieval layer", "keep scope minimal", "toy data is fine"]}
        events2, lat2, err2 = post_ndjson(base, rpath, payload2, token, timeout)
        latency_ms += lat2
        resumes += 1
        if err2:
            return {"id": item.id, "type": "run", "error": f"resume: {err2}", "latency_ms": round(latency_ms), "repo_id": repo_id, "interrupts": len(interrupts), "resumes": resumes}
        events = events + events2
    interrupts = [e for e in events if isinstance(e, dict) and e.get("event") == "interrupt"]
    result = next(
        (
            (e.get("data") or e.get("result") or e)
            for e in reversed(events)
            if isinstance(e, dict) and e.get("event") == "result"
        ),
        None,
    )
    if not isinstance(result, dict):
        return {"id": item.id, "type": "run", "error": "no run/result event", "latency_ms": round(latency_ms), "repo_id": repo_id, "interrupts": len(interrupts), "resumes": resumes, "n_events": len(events)}
    evaluation = result.get("evaluation") or {}
    usage = result.get("usage") or {}
    integration = result.get("integration")
    poc = result.get("poc") or {}
    sandbox = poc.get("sandbox") or {}
    return {
        "id": item.id,
        "type": "run",
        "repo_id": repo_id,
        "status": result.get("status"),
        "verdict": result.get("verdict") or evaluation.get("verdict"),
        "sandbox_passed": evaluation.get("sandbox_passed", sandbox.get("passed")),
        "worthiness": (evaluation.get("worthiness") or {}).get("worthiness") if isinstance(evaluation.get("worthiness"), dict) else None,
        "llm_calls": usage.get("calls", evaluation.get("llm_calls")),
        "cost_usd": usage.get("cost_usd", evaluation.get("cost_usd")),
        "wall_ms": evaluation.get("wall_ms"),
        "demo_runtime_ms": (evaluation.get("performance") or {}).get("demo_runtime_ms") if isinstance(evaluation.get("performance"), dict) else None,
        "memory_peak_kb": (evaluation.get("performance") or {}).get("memory_peak_kb") if isinstance(evaluation.get("performance"), dict) else None,
        "integration": "proposed" if integration else "skipped",
        "interrupts": len(interrupts),
        "resumes": resumes,
        "n_events": len(events),
        "latency_ms": round(latency_ms),
    }


def pct(values: list[float]) -> dict:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return {"n": 0}
    vals.sort()
    out: dict = {"n": len(vals), "mean": round(statistics.mean(vals), 2)}
    out["p50"] = round(vals[len(vals) // 2], 2)
    if len(vals) >= 4:
        qs = statistics.quantiles(vals, n=100)
        out["p95"] = round(qs[94], 2)
    else:
        out["p95"] = round(max(vals), 2)
    out["min"] = round(min(vals), 2)
    out["max"] = round(max(vals), 2)
    return out


def aggregate(items: list[dict]) -> dict:
    def _sel(t: str) -> list[dict]:
        return [d for d in items if d.get("type") == t and not d.get("error")]

    faith = _sel("faithfulness")
    code = _sel("code")
    feas = _sel("feasibility")
    runs = _sel("run")
    agg: dict = {
        "faithfulness": {
            "n": len(faith),
            "recall": round(sum(1 for d in faith if d.get("recall_hit")) / len(faith), 3) if faith else None,
            "mrr_mean": round(statistics.mean([d.get("mrr", 0) for d in faith]), 3) if faith else None,
            "citation_pass_rate": round(sum(1 for d in faith if d.get("citation_pass")) / len(faith), 3) if faith else None,
            "acceptance_rate": round(sum(1 for d in faith if d.get("acceptance_hit")) / len([d for d in faith if d.get("acceptance_hit") is not None]), 3) if any(d.get("acceptance_hit") is not None for d in faith) else None,
            "latency_ms": pct([d.get("latency_ms", 0) for d in faith]),
        },
        "code": {
            "n": len(code),
            "sandbox_pass_rate": round(sum(1 for d in code if d.get("sandbox_passed")) / len(code), 3) if code else None,
            "demo_runtime_ms": pct([d.get("demo_runtime_ms", 0) for d in code if isinstance(d.get("demo_runtime_ms"), (int, float))]),
            "latency_ms": pct([d.get("latency_ms", 0) for d in code]),
        },
        "feasibility": {
            "n": len(feas),
            "verdict_accuracy": round(sum(1 for d in feas if d.get("verdict_match")) / len([d for d in feas if d.get("verdict_match") is not None]), 3) if any(d.get("verdict_match") is not None for d in feas) else None,
            "verdict_lenient": round(sum(1 for d in feas if d.get("verdict_match_lenient")) / len([d for d in feas if d.get("verdict_match_lenient") is not None]), 3) if any(d.get("verdict_match_lenient") is not None for d in feas) else None,
            "unsupported_rate": round(sum(d.get("n_unsupported", 0) for d in feas) / max(sum(d.get("n_claims", 0) for d in feas), 1), 3) if feas else None,
            "latency_ms": pct([d.get("latency_ms", 0) for d in feas]),
        },
        "run": {
            "n": len(runs),
            "done_rate": round(sum(1 for d in runs if d.get("status") == "done") / len(runs), 3) if runs else None,
            "sandbox_pass_rate": round(sum(1 for d in runs if d.get("sandbox_passed")) / len(runs), 3) if runs else None,
            "total_cost_usd": round(sum(d.get("cost_usd", 0) or 0 for d in runs), 6) if runs else 0,
            "total_llm_calls": sum(d.get("llm_calls", 0) or 0 for d in runs) if runs else 0,
            "latency_ms": pct([d.get("latency_ms", 0) for d in runs]),
        },
        "errors": sum(1 for d in items if d.get("error")),
    }
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description="Run live eval against a running ProdRag stack")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--golden", default="eval/golden/v1")
    ap.add_argument("--out", default="eval/results/live.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--timeout-query", type=int, default=60)
    ap.add_argument("--timeout-code", type=int, default=180)
    ap.add_argument("--timeout-feas", type=int, default=300)
    ap.add_argument("--timeout-run", type=int, default=900)
    ap.add_argument("--max-llm-calls", type=int, default=25)
    ap.add_argument("--max-wall-s", type=int, default=900)
    ap.add_argument("--baseline", choices=["off", "llm-only"], default="off")
    ap.add_argument("--repo-map", default="{}", help="JSON alias->repo_id, e.g. '{\"self\":\"abc123\"}'")
    ap.add_argument("--only", default="", help="Comma list subset of faithfulness,code,feasibility,run")
    ap.add_argument("--limit", type=int, default=0, help="Max items per type (0 = all)")
    ap.add_argument("--ids", default="", help="Comma list of item ids to run (overrides --only/--limit)")
    args = ap.parse_args()

    token = os.environ.get("PRODRAG_API_TOKEN", "")
    if not token:
        print("PRODRAG_API_TOKEN is required", file=sys.stderr)
        sys.exit(2)
    try:
        repo_map = json.loads(args.repo_map)
    except Exception as e:
        print(f"bad --repo-map: {e}", file=sys.stderr)
        sys.exit(2)

    baseline = args.baseline == "llm-only"
    k = 1 if baseline else args.k
    only = {t.strip() for t in args.only.split(",") if t.strip()}
    ids = {t.strip() for t in args.ids.split(",") if t.strip()}

    items = load_golden(args.golden)
    if ids:
        items = [it for it in items if it.id in ids]
    if only:
        items = [it for it in items if it.type in only]
    if args.limit:
        by_type: dict[str, list] = {}
        for it in items:
            by_type.setdefault(it.type, []).append(it)
        items = [it for t in by_type for it in by_type[t][: args.limit]]

    health = get_json(args.base_url, "/health", token)
    search_status = get_json(args.base_url, "/tools/search/status", token)
    config = {
        "base_url": args.base_url,
        "golden": str(args.golden),
        "n_items": len(items),
        "k": k,
        "baseline": args.baseline,
        "max_llm_calls": args.max_llm_calls,
        "max_wall_s": args.max_wall_s,
        "repo_map": repo_map,
        "health": health.get("body"),
        "search": search_status.get("body"),
        "started_at": time.time(),
    }
    print(f"live eval: {len(items)} items vs {args.base_url} (baseline={args.baseline} k={k})")
    print(f"health={health.get('body')} search={search_status.get('body')}")

    results: list[dict] = []
    for n, item in enumerate(items, 1):
        t0 = time.time()
        try:
            if item.type == "faithfulness":
                res = eval_faith(args.base_url, item, token, k, args.timeout_query)
            elif item.type == "code":
                res = eval_code(args.base_url, item, token, args.timeout_code)
            elif item.type == "feasibility":
                res = eval_feasibility(args.base_url, item, token, repo_map, k, args.timeout_feas)
            elif item.type == "run":
                res = eval_run(args.base_url, item, token, repo_map, k, args.max_llm_calls, args.max_wall_s, args.timeout_run)
            else:
                res = {"id": item.id, "type": item.type, "error": f"unknown type {item.type}"}
        except Exception as e:
            res = {"id": item.id, "type": item.type, "error": f"harness: {e}"[:500]}
        res["baseline"] = baseline
        res["wall_s"] = round(time.time() - t0, 1)
        results.append(res)
        status = res.get("error") or f"{res.get('verdict', res.get('sandbox_passed', res.get('recall_hit', 'ok')))}"
        print(f"[{n}/{len(items)}] {item.id} ({item.type}) -> {status} ({res['wall_s']}s)")

    out = {
        "config": config,
        "aggregate": aggregate(results),
        "items": results,
        "finished_at": time.time(),
        "elapsed_s": round(time.time() - config["started_at"], 1),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path} ({len(results)} items, {out['elapsed_s']}s)")
    print(json.dumps(out["aggregate"], indent=2))


if __name__ == "__main__":
    main()
