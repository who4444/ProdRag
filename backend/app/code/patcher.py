"""Extend-mode patcher — apply ExtendFileTask changes to a tree directory.

create = write new file, modify = replace whole file content, or apply
search_replace blocks (exact matches, all occurrences). Never raises.
"""

import logging
import time
from pathlib import Path

from .schemas import ExtendFileTask

logger = logging.getLogger(__name__)


def _safe_rel(path: str) -> str | None:
    rel = (path or "").strip().lstrip("/")
    if not rel or rel.startswith("..") or "../" in rel or rel.startswith("~"):
        return None
    return rel


def apply_changes(root: Path, tasks: list[ExtendFileTask]) -> tuple[list[dict], str]:
    """Apply create/modify tasks under root.

    Returns (changes: [{path, action, applied, bytes}], log).
    """
    changes: list[dict] = []
    log_lines: list[str] = []
    t0 = time.perf_counter()
    for t in tasks:
        rel = _safe_rel(t.path)
        if rel is None:
            log_lines.append(f"skip unsafe path: {t.path!r}")
            changes.append({"path": t.path, "action": t.action, "applied": False, "bytes": 0})
            continue
        target = root / rel
        try:
            action = t.action
            if action == "modify" and not target.exists():
                log_lines.append(f"modify target missing, treated as create: {rel}")
                action = "create"
            if action == "modify" and not t.content and not t.search_replace:
                # Never clobber an existing file with empty content
                changes.append({"path": rel, "action": "modify", "applied": False, "bytes": 0, "error": "empty content"})
                log_lines.append(f"skip empty modify: {rel}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if action == "modify" and t.search_replace and not t.content:
                text = target.read_text(encoding="utf-8", errors="replace")
                applied = 0
                for block in t.search_replace:
                    s = str(block.get("search", ""))
                    if not s:
                        continue
                    if s in text:
                        text = text.replace(s, str(block.get("replace", "")))
                        applied += 1
                    else:
                        log_lines.append(f"search block miss in {rel}: {s[:60]!r}")
                if applied:
                    target.write_text(text, encoding="utf-8")
                    changes.append({"path": rel, "action": "modify", "applied": True, "bytes": len(text)})
                    log_lines.append(f"modified {rel} ({applied} replacement(s))")
                else:
                    changes.append({"path": rel, "action": t.action, "applied": False, "bytes": 0})
                    log_lines.append(f"no search block applied and no full content for {rel}")
            else:
                # create, or whole-file modify (content is the full replacement)
                content = t.content or ""
                existed = target.exists()
                target.write_text(content, encoding="utf-8")
                final_action = "modify" if (existed and action == "modify") else "create"
                changes.append({
                    "path": rel,
                    "action": final_action,
                    "applied": True,
                    "bytes": len(content),
                })
                log_lines.append(f"wrote {rel} ({'overwrite' if existed else 'new'}, {len(content)} bytes)")
        except Exception as e:
            logger.exception("patch failed for %s", rel)
            changes.append({"path": rel, "action": t.action, "applied": False, "bytes": 0, "error": str(e)[:200]})
            log_lines.append(f"error {rel}: {e}")
    log = "; ".join(log_lines) or "no changes"
    logger.info("patcher: %s in %.2fs", log, time.perf_counter() - t0)
    return changes, log
