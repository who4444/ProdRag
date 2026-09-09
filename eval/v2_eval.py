"""v2 eval: faithfulness (recall + citation) + code sandbox pass rate. Uses v2_golden.jsonl (10 items: 5 faithfulness + 5 code)."""

import json
import os
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
GOLDEN = sys.argv[2] if len(sys.argv) > 2 else "eval/v2_golden.jsonl"
TOKEN = os.environ.get("PRODRAG_API_TOKEN", "")

faith_hits = faith_total = 0
code_pass = code_total = 0

import re

SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
CITE_RE = re.compile(r"\[(\d+)\]")

cite_hits = cite_total = 0
accept_hits = accept_total = 0

for line in open(GOLDEN):
    item = json.loads(line)
    if item["type"] == "faithfulness":
        req = urllib.request.Request(
            f"{BASE}/query",
            data=json.dumps({"question": item["question"], "k": 5}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
        )
        try:
            stream = urllib.request.urlopen(req, timeout=30).read().decode()
            lines = stream.splitlines()
            sources = json.loads(lines[0])["items"]
            # join text deltas for citation check
            text = "".join(json.loads(l).get("delta", "") for l in lines[1:] if l.strip().startswith("{"))
            faith_total += 1
            if any(s["source"] in item["expected_sources"] for s in sources):
                faith_hits += 1
            # citation coverage: cited_claims / total_claims
            sents = [s.strip() for s in SENT_RE.split(text.strip()) if s.strip()]
            if sents:
                cite_total += 1
                cited = sum(1 for s in sents if any(1 <= int(n) <= len(sources) for n in CITE_RE.findall(s)))
                # consider cited if at least half sentences have citations
                if cited / len(sents) >= 0.5:
                    cite_hits += 1
            # acceptance strings
            accept_total += 1
            if all(a.lower() in text.lower() for a in item.get("acceptance", [])):
                accept_hits += 1
        except Exception as e:
            print(f"faith {item['id']} error: {e}")
            faith_total += 1
            cite_total += 1
            accept_total += 1
    elif item["type"] == "code":
        # code: artifact -> code/generate, check sandbox.passed
        import time

        # 1. create minimal artifact from item
        artifact = {
            "requirements": item.get("requirements", {"goal": item.get("idea", "demo"), "core_idea": item.get("idea", "demo"), "demo_type": "script", "constraints": [], "audience": "researcher", "open_questions": []}),
            "summaries": [
                {"direction_id": "core_algorithm", "findings": item.get("requirements", {}).get("core_idea", ""), "sources": [], "confidence": "high", "gaps": []},
                {"direction_id": "evaluation", "findings": "; ".join(item.get("acceptance_criteria", [])), "sources": [], "confidence": "high", "gaps": []},
            ],
            "validation": {"coverage": {"core_algorithm": True}, "missing": [], "feasible": True, "risks": []},
            "demo_spec": {"entrypoint": "demo.py", "tech_stack": ["python"] + item.get("expected_deps", []), "core_algorithm": item.get("requirements", {}).get("core_idea", ""), "pseudocode": "", "acceptance_criteria": item.get("acceptance_criteria", ["demo.py exits 0"])},
            "sources": [],
            "created_at": time.time(),
        }
        try:
            req = urllib.request.Request(
                f"{BASE}/code/generate",
                data=json.dumps({"artifact": artifact}).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
            )
            stream = urllib.request.urlopen(req, timeout=120).read().decode()
            # last line is code/artifact with sandbox
            last = [json.loads(l) for l in stream.splitlines() if l.strip()][-1]
            data = last.get("data") or last
            sandbox = data.get("sandbox") or {}
            if not sandbox:
                # try nested code_artifact
                sandbox = data.get("code_artifact", {}).get("sandbox", {})
            code_total += 1
            if sandbox.get("passed"):
                code_pass += 1
            else:
                print(f"code {item['id']} sandbox failed: {sandbox.get('pytest_log','')[:200]}")
        except Exception as e:
            print(f"code {item['id']} error: {e}")
            code_total += 1

if faith_total:
    print(f"faithfulness recall: {faith_hits}/{faith_total} = {faith_hits/faith_total:.0%}")
    print(f"citation coverage (>=50% sents cited): {cite_hits}/{cite_total} = {cite_hits/cite_total:.0%}" if cite_total else "")
    print(f"acceptance criteria: {accept_hits}/{accept_total} = {accept_hits/accept_total:.0%}" if accept_total else "")
if code_total:
    print(f"code sandbox pass rate: {code_pass}/{code_total} = {code_pass/code_total:.0%}")
