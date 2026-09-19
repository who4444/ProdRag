"""Citation gate — deterministic validation of evidence against the repo tree.

The LLM proposes evidence; only the filesystem decides. A claim becomes
`supported` only when at least one ref resolves to a real path and line range.
This is what makes "plausible" mean evidence-backed rather than asserted.
"""

import hashlib
import logging
from pathlib import Path

from .schemas import EvidenceRef, FeasibilityClaim

logger = logging.getLogger(__name__)

SNIPPET_MAX = 500


def _line_count(path: Path) -> int | None:
    try:
        return len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return None


def _read_lines(path: Path, start: int | None, end: int | None) -> str:
    if start is None:
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    stop = end if end is not None else start
    return "\n".join(lines[start - 1 : stop])[:SNIPPET_MAX]


def validate_ref(root: Path, ref: EvidenceRef) -> EvidenceRef:
    """Set ref.valid/ref.reason in place. Never raises."""
    root = Path(root).resolve()
    raw = (ref.path or "").strip().lstrip("/")
    ref.valid = False
    ref.reason = ""
    if not raw:
        ref.reason = "empty path"
        return ref
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        ref.reason = "path escapes repo root"
        return ref
    if not candidate.is_file():
        ref.reason = "file not found"
        return ref
    total = _line_count(candidate)
    if total is None:
        ref.reason = "unreadable file"
        return ref
    if ref.start_line is not None and not (1 <= ref.start_line <= total):
        ref.reason = f"start_line {ref.start_line} outside 1..{total}"
        return ref
    if ref.end_line is not None and not (1 <= ref.end_line <= total):
        ref.reason = f"end_line {ref.end_line} outside 1..{total}"
        return ref
    if ref.start_line is not None and ref.end_line is not None and ref.end_line < ref.start_line:
        ref.reason = "end_line before start_line"
        return ref
    ref.valid = True
    if not ref.snippet:
        ref.snippet = _read_lines(candidate, ref.start_line, ref.end_line)
    return ref


def apply_citation_gate(root: Path, claims: list[FeasibilityClaim]) -> list[FeasibilityClaim]:
    for claim in claims:
        for ref in claim.evidence:
            validate_ref(root, ref)
        if claim.provisional:
            claim.status = "unsupported"
        elif not claim.evidence:
            claim.status = "assumption"
        elif any(r.valid for r in claim.evidence):
            claim.status = "supported"
        else:
            claim.status = "unsupported"
        if not claim.id:
            digest = hashlib.md5(claim.statement.encode()).hexdigest()[:8]
            claim.id = f"{claim.dimension}-{digest}"
    return claims
