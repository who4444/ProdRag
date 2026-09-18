"""Sparse (BM25) vector extraction for Qdrant hybrid search.

Qdrant computes IDF itself from the stored sparse vectors (modifier=IDF), so
we only supply raw term frequencies. Indices are a deterministic 32-bit hash
of the lowercased word (Qdrant's sparse index space) so index-time and
query-time vocabularies always match across processes.

ponytail: tokenizer is naive (lowercase alnum runs, stopword list, no
stemming) — swap for a real tokenizer when eval shows recall loss.
"""

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Closed-class English words carry little retrieval signal but high term
# frequency (a long chunk full of "is/the/where" outranks the actual subject).
_STOPWORDS = frozenset(
    "a an and are as at be but by for from had has have how i if in into is it its "
    "me my no not of on or our so than that the their them then there these this to "
    "was we were what when where which who why will with you your".split()
)


def tokens(text: str, stopwords: bool = True) -> list[str]:
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if stopwords and tok in _STOPWORDS:
            continue
        out.append(tok)
    return out


def to_sparse(text: str, stopwords: bool = True) -> dict:
    """Term-frequency sparse vector for a text: {"indices": [...], "values": [...]}.

    Weights are log-tf saturated: Qdrant's sparse scorer applies IDF but has no
    BM25 tf saturation/length normalization, so a chunk repeating one term would
    otherwise outrank the chunk that actually covers the query.
    """
    counts: dict[str, int] = {}
    for tok in tokens(text, stopwords):
        counts[tok] = counts.get(tok, 0) + 1
    by_index: dict[int, float] = {}
    for tok, n in counts.items():
        idx = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)  # u32: Qdrant sparse index space
        by_index[idx] = by_index.get(idx, 0.0) + 1.0 + math.log(n)
    indices = sorted(by_index)
    return {"indices": indices, "values": [round(by_index[i], 6) for i in indices]}