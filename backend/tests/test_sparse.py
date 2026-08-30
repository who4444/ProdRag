from app.core.sparse import to_sparse
from app.core.vectorstore import _hybrid_prefetch


def test_to_sparse_deterministic_and_sorted():
    s1 = to_sparse("The cat and the cat and dog")
    s2 = to_sparse("The cat and the cat and dog")
    assert s1 == s2
    assert s1["indices"] == sorted(s1["indices"])
    assert len(s1["indices"]) == len(s1["values"])


def test_to_sparse_counts_terms():
    s = to_sparse("cat cat dog")
    assert len(s["indices"]) == 2
    assert set(s["values"]) == {1.0, 2.0}


def test_to_sparse_case_and_punct_insensitive():
    assert to_sparse("Anomaly detection!") == to_sparse("anomaly detection")


def test_to_sparse_repeated_terms_accumulate():
    s = to_sparse("cat cat")
    assert len(s["indices"]) == 1
    assert s["values"] == [2.0]


def test_hybrid_prefetch_uses_dense_and_bm25():
    p = _hybrid_prefetch([0.1, 0.2], {"indices": [1, 2], "values": [1.0, 2.0]}, None, 12)
    assert len(p) == 2
    assert p[0].using is None
    assert p[1].using == "bm25"
    assert p[0].limit == 12 == p[1].limit