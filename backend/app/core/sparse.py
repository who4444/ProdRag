"""Sparse (BM25) vector extraction for Qdrant hybrid search.

Qdrant computes IDF itself from the stored sparse vectors (modifier=IDF), so
we only supply raw term frequencies. Indices are a deterministic 32-bit hash
of the lowercased word (Qdrant's sparse index space) so index-time and
query-time vocabularies always match across processes.

ponytail: tokenizer is naive (lowercase alnum runs, no stemming/stopwords) —
IDF already downweights corpus-common words. Swap for a real tokenizer when
eval shows recall loss on the actual corpus.
"""

import hashlib
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def to_sparse(text: str) -> dict:
    """Term-frequency sparse vector for a text: {"indices": [...], "values": [...]}."""
    counts: dict[str, int] = {}
    for tok in _TOKEN_RE.findall(text.lower()):
        counts[tok] = counts.get(tok, 0) + 1
    by_index: dict[int, int] = {}
    for tok, n in counts.items():
        idx = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)  # u32: Qdrant sparse index space
        by_index[idx] = by_index.get(idx, 0) + n  # hash collision -> merge counts
    indices = sorted(by_index)
    return {"indices": indices, "values": [float(by_index[i]) for i in indices]}