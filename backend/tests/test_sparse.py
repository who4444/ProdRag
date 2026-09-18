from app.core.sparse import to_sparse
from app.core.vectorstore import _hybrid_prefetch


def test_to_sparse_deterministic_and_sorted():
    s1 = to_sparse("The cat and the cat and dog")
    s2 = to_sparse("The cat and the cat and dog")
    assert s1 == s2
    assert s1["indices"] == sorted(s1["indices"])
    assert len(s1["indices"]) == len(s1["values"])


def test_to_sparse_counts_terms_with_log_tf_saturation():
    import math

    s = to_sparse("cat cat dog")
    assert len(s["indices"]) == 2
    values = sorted(s["values"])
    assert values[0] == 1.0
    assert values[1] == round(1.0 + math.log(2), 6)


def test_to_sparse_case_and_punct_insensitive():
    assert to_sparse("Anomaly detection!") == to_sparse("anomaly detection")


def test_to_sparse_repeated_terms_accumulate_sublinearly():
    import math

    s = to_sparse("cat cat")
    assert len(s["indices"]) == 1
    assert s["values"] == [round(1.0 + math.log(2), 6)]


def test_to_sparse_drops_stopwords_by_default():
    assert len(to_sparse("what is the sandbox")["indices"]) == 1
    assert len(to_sparse("what is the sandbox", stopwords=False)["indices"]) == 4


def test_hybrid_prefetch_uses_dense_and_bm25():
    p = _hybrid_prefetch([0.1, 0.2], {"indices": [1, 2], "values": [1.0, 2.0]}, None, 12)
    assert len(p) == 2
    assert p[0].using is None
    assert p[1].using == "bm25"
    assert p[0].limit == 12 == p[1].limit