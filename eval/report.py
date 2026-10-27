"""Render eval results JSON as a markdown table + summary.

Usage:
  uv run python eval/report.py eval/results/live-v1.json [--out eval/results/SUMMARY.md]
  uv run python eval/report.py eval/results/live-v1.json eval/results/baseline-v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(v, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}" if v < 10 else f"{v:.2f}"
    return str(v)


def lat(d: dict) -> str:
    if not d or not d.get("n"):
        return "-"
    return f"p50 {d.get('p50')} / p95 {d.get('p95')} ms (n={d.get('n')})"


def render(path: str) -> tuple[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg, agg = data.get("config", {}), data.get("aggregate", {})
    f, c, fe, r = agg.get("faithfulness", {}), agg.get("code", {}), agg.get("feasibility", {}), agg.get("run", {})
    lines = [
        f"## Eval results — {Path(path).name}",
        "",
        f"- base_url={cfg.get('base_url')} golden={cfg.get('golden')} baseline={cfg.get('baseline')} k={cfg.get('k')}",
        f"- items={cfg.get('n_items')} elapsed={data.get('elapsed_s')}s errors={agg.get('errors')}",
        f"- health={cfg.get('health')} search={cfg.get('search')}",
        "",
        "| Layer | n | Quality | Latency |",
        "|---|---|---|---|",
        f"| RAG /query | {f.get('n')} | recall {fmt(f.get('recall'))} · MRR {fmt(f.get('mrr_mean'))} · cite≥50% {fmt(f.get('citation_pass_rate'))} · accept {fmt(f.get('acceptance_rate'))} | {lat(f.get('latency_ms', {}))} |",
        f"| Code /code/generate | {c.get('n')} | sandbox.passed {fmt(c.get('sandbox_pass_rate'))} | {lat(c.get('latency_ms', {}))} |",
        f"| Feasibility /feasibility/analyze | {fe.get('n')} | verdict_acc {fmt(fe.get('verdict_accuracy'))} · lenient {fmt(fe.get('verdict_lenient'))} · unsupported {fmt(fe.get('unsupported_rate'))} | {lat(fe.get('latency_ms', {}))} |",
        f"| Runs /runs e2e | {r.get('n')} | done {fmt(r.get('done_rate'))} · sandbox {fmt(r.get('sandbox_pass_rate'))} · calls {r.get('total_llm_calls')} · cost ${fmt(r.get('total_cost_usd'), 4)} | {lat(r.get('latency_ms', {}))} |",
        "",
    ]
    # Per-item failures for triage.
    fails = [it for it in data.get("items", []) if it.get("error")]
    if fails:
        lines += ["### Errors", ""]
        lines += [f"- {it.get('id')} ({it.get('type')}): {it.get('error')}"[:300] for it in fails[:20]]
        lines += [""]
    return "\n".join(lines), data


def main() -> None:
    ap = argparse.ArgumentParser(description="Render eval results as markdown")
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    blocks = []
    for p in args.results:
        md, _ = render(p)
        blocks.append(md)
    if len(args.results) == 2:
        a = json.loads(Path(args.results[0]).read_text(encoding="utf-8"))["aggregate"]
        b = json.loads(Path(args.results[1]).read_text(encoding="utf-8"))["aggregate"]
        blocks.append("### Workflow vs baseline (live − baseline)")
        for layer in ("faithfulness", "code", "feasibility", "run"):
            da, db = a.get(layer, {}), b.get(layer, {})
            key = {"faithfulness": "recall", "code": "sandbox_pass_rate", "feasibility": "verdict_accuracy", "run": "done_rate"}[layer]
            va, vb = da.get(key), db.get(key)
            delta = "n/a" if va is None or vb is None else f"{va - vb:+.3f}"
            blocks.append(f"- {layer} {key}: live={va} baseline={vb} delta={delta}")
        blocks.append("")
    text = "\n".join(blocks)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
