"""Citation gate: only the filesystem decides (M2)."""

from app.feasibility.citation import apply_citation_gate, validate_ref
from app.feasibility.schemas import EvidenceRef, FeasibilityClaim


def _repo(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    return tmp_path


def test_validate_ref_accepts_real_file_and_lines(tmp_path):
    root = _repo(tmp_path)
    ref = validate_ref(root, EvidenceRef(path="pkg/mod.py", start_line=2, end_line=3))
    assert ref.valid is True
    assert ref.snippet == "b = 2\nc = 3"


def test_validate_ref_rejects_missing_escape_and_bad_lines(tmp_path):
    root = _repo(tmp_path)
    assert validate_ref(root, EvidenceRef(path="pkg/nope.py")).reason == "file not found"
    assert "escapes" in validate_ref(root, EvidenceRef(path="../outside.py")).reason
    assert "outside" in validate_ref(root, EvidenceRef(path="pkg/mod.py", start_line=99)).reason
    assert "outside" in validate_ref(root, EvidenceRef(path="pkg/mod.py", start_line=1, end_line=99)).reason
    assert validate_ref(root, EvidenceRef(path="pkg/mod.py", start_line=3, end_line=1)).reason == "end_line before start_line"
    assert validate_ref(root, EvidenceRef(path="")).reason == "empty path"


def test_gate_sets_status_from_evidence(tmp_path):
    root = _repo(tmp_path)
    supported = FeasibilityClaim(
        dimension="technical",
        statement="fits",
        evidence=[EvidenceRef(path="pkg/mod.py", start_line=1)],
    )
    unsupported = FeasibilityClaim(
        dimension="integration",
        statement="claim",
        evidence=[EvidenceRef(path="ghost.py", start_line=1)],
    )
    assumption = FeasibilityClaim(dimension="data", statement="we assume toy data")

    claims = apply_citation_gate(root, [supported, unsupported, assumption])

    assert [c.status for c in claims] == ["supported", "unsupported", "assumption"]
    assert all(c.id for c in claims)  # deterministic ids assigned


def test_gate_requires_code_evidence_for_technical_and_integration(tmp_path):
    """Docs describe the system; they are not implementation evidence."""
    root = _repo(tmp_path)
    (root / "PROJECT.md").write_text("mod.py implements the feature\n", encoding="utf-8")

    doc_technical = FeasibilityClaim(
        dimension="technical",
        statement="PROJECT.md says it exists",
        evidence=[EvidenceRef(path="PROJECT.md", start_line=1)],
    )
    doc_data = FeasibilityClaim(
        dimension="data",
        statement="data described in docs",
        evidence=[EvidenceRef(path="PROJECT.md", start_line=1)],
    )
    code_plus_doc = FeasibilityClaim(
        dimension="technical",
        statement="implemented in code",
        evidence=[EvidenceRef(path="pkg/mod.py", start_line=1), EvidenceRef(path="PROJECT.md", start_line=1)],
    )

    apply_citation_gate(root, [doc_technical, doc_data, code_plus_doc])

    assert doc_technical.status == "assumption"  # downgraded: docs only
    assert doc_data.status == "supported"  # data may be documented
    assert code_plus_doc.status == "supported"  # has a code ref


def test_gate_treats_provisional_claims_as_unsupported(tmp_path):
    root = _repo(tmp_path)
    provisional = FeasibilityClaim(
        dimension="technical",
        statement="Assessment unavailable (LLM failure); raw retrieval attached.",
        evidence=[EvidenceRef(path="pkg/mod.py", start_line=1, end_line=3)],
        provisional=True,
    )
    apply_citation_gate(root, [provisional])
    assert provisional.status == "unsupported"
    assert provisional.evidence[0].valid is True  # evidence resolves, but the claim is not evidence-backed
