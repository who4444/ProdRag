"""Recall@k against a running /query. Usage: PRODRAG_API_TOKEN=... python eval/recall.py [base_url] [golden] [k]"""

import json
import os
import sys
import urllib.request


def recall_at_k(base_url, path, k, token):
    hits = total = 0
    for line in open(path):
        item = json.loads(line)
        req = urllib.request.Request(
            base_url + "/query",
            data=json.dumps({"question": item["question"], "k": k}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )
        stream = urllib.request.urlopen(req).read().decode()
        sources = json.loads(stream.splitlines()[0])["items"]
        total += 1
        hits += any(s["source"] in item["expected_sources"] for s in sources)
    print(f"recall@{k}: {hits}/{total} = {hits / total:.0%}")


if __name__ == "__main__":
    recall_at_k(
        sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000",
        sys.argv[2] if len(sys.argv) > 2 else "eval/golden.jsonl",
        int(sys.argv[3]) if len(sys.argv) > 3 else 5,
        os.environ.get("PRODRAG_API_TOKEN", ""),
    )
