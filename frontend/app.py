"""ProdRag UI — single-page full run: idea -> research -> code (demo + extend PoC).

Run:
    uv sync --extra frontend
    cd frontend && uv run streamlit run app.py
"""

import json
import os

import pandas as pd
import streamlit as st

from api import (
    APIError,
    code_extend_stream,
    code_generate_stream,
    document_delete,
    document_reingest,
    document_status,
    feasibility_stream,
    get_code_run,
    get_conversation_messages,
    get_file,
    get_rd_run,
    health,
    list_code_runs,
    list_conversations,
    list_documents,
    list_rd_runs,
    memory_episodes,
    memory_search,
    query_stream,
    rd_analyze,
    rd_research_stream,
    repo_index,
    repo_list,
    upload_document,
    web_search_probe,
)

st.set_page_config(page_title="ProdRag", page_icon="📚", layout="wide")


def _default(name: str, fallback: str) -> str:
    try:
        return st.secrets.get(name, os.environ.get(name, fallback))
    except Exception:
        return os.environ.get(name, fallback)


# ----------------------------------------------------------------------------
# Run-scoped session state: every stage stores its result here so it survives
# widget reruns. The one-shot chain flags are popped exactly once, which is
# what keeps long streams from re-executing on every later rerun.
# ----------------------------------------------------------------------------
_SS_DEFAULTS = {
    "idea": "",
    "mode": "demo",
    "auto": False,
    "req": None,
    "questions": None,
    "artifact": None,
    "rd_run_id": None,
    "feas": None,
    "feas_pre": False,
    "repo_id": "",
    "demo": None,
    "ext": None,
}

_CHAIN_FLAGS = ("_do_research", "_do_code", "_do_extend", "_do_feas")


def _ss() -> None:
    for key, value in _SS_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def _clear_run() -> None:
    """Reset every run-scoped key (sidebar Reset + section-1 Clear)."""
    for key, value in _SS_DEFAULTS.items():
        st.session_state[key] = value
    for flag in _CHAIN_FLAGS:
        st.session_state.pop(flag, None)


def _sandbox_status(run: dict | None) -> str:
    sandbox = (run or {}).get("sandbox") or {}
    if not sandbox:
        return "—"
    return "✅ passed" if sandbox.get("passed") else "❌ failed"


_ss()


# ----------------------------------------------------------------------------
# Sidebar: connection settings + health/search probes + full run reset.
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 ProdRag")
    api_url = st.text_input(
        "API URL", value=_default("PRODRAG_API_URL", "http://localhost:8000")
    )
    api_token = st.text_input(
        "API token", type="password", value=_default("PRODRAG_API_TOKEN", "change-me")
    )
    if st.button("Check connection"):
        try:
            health(api_url)
            st.success("API reachable")
        except Exception as exc:
            st.error(f"Unreachable: {exc}")
    if st.button("Search status"):
        try:
            res = web_search_probe(api_url, api_token, "test", k=1)
            st.json(res)
        except Exception as exc:
            st.error(f"Search probe failed: {exc}")
    st.caption("TTL: research 30d / code 30d / episodes 7d / conv 1d (lazy)")
    st.divider()
    if st.button("Reset run"):
        _clear_run()
        st.rerun()


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("### 🚀 Full run — idea → research → code")
st.caption(
    "One page, one click: clarify → research → artifact → sandboxed PoC "
    "(demo and/or codebase extension)."
)


# ----------------------------------------------------------------------------
# 1 · Idea — capture inputs; one click resets downstream state and chains on.
# ----------------------------------------------------------------------------
st.markdown("### 1 · Idea")
idea = st.text_area(
    "Idea / paper",
    key="idea_widget",
    placeholder="e.g. Turn Attention Is All You Need into a minimal demo on toy data",
)
mode = st.radio(
    "Code stage",
    ["demo", "extend", "both"],
    horizontal=True,
    format_func=lambda m: {
        "demo": "Demo PoC (from research artifact)",
        "extend": "Extend a codebase (PoC patch)",
        "both": "Both",
    }[m],
    help="demo: /code/generate on the research artifact. extend: /code/extend against an indexed/local codebase.",
)
feas_pre = st.checkbox(
    "Feasibility pre-check before extend",
    value=False,
    help="/feasibility/analyze with the selected repo; verdict shown before the extend run.",
)
col_run, col_clear = st.columns(2)
if col_run.button("Run full pipeline", type="primary", disabled=not idea.strip()):
    # Full-run reset: nothing from a previous chain may leak into this one.
    st.session_state.mode = mode
    st.session_state.idea = idea
    st.session_state.feas_pre = feas_pre
    st.session_state.auto = True
    st.session_state.req = None
    st.session_state.questions = None
    st.session_state.artifact = None
    st.session_state.demo = None
    st.session_state.ext = None
    st.session_state.feas = None
    st.session_state.rd_run_id = None
    try:
        res = rd_analyze(api_url, api_token, idea)
    except APIError as exc:
        st.error(f"Analyze failed: {exc}")
    else:
        if res.get("ready"):
            st.session_state.req = res["requirements"]
            st.session_state["_do_research"] = True  # keep the one-click chain moving
        else:
            st.session_state.questions = res.get("questions", [])
        st.rerun()
if col_clear.button("Clear"):
    _clear_run()
    st.rerun()


# ----------------------------------------------------------------------------
# 2 · Requirements — analyzer clarification handshake (+ manual Analyze only).
# ----------------------------------------------------------------------------
st.markdown("### 2 · Requirements")
if st.session_state.questions:
    st.warning("Clarify before research:")
    answers = [
        st.text_input(f"Q{i+1}: {q}", key=f"ans_{i}")
        for i, q in enumerate(st.session_state.questions)
    ]
    if st.button("Continue with answers", type="primary", key="answers_btn"):
        st.session_state.idea = idea
        try:
            res = rd_analyze(
                api_url, api_token, idea, answers=[a for a in answers if a.strip()]
            )
        except APIError as exc:
            st.error(f"Analyze failed: {exc}")
        else:
            if res.get("ready"):
                st.session_state.req = res["requirements"]
                st.session_state.questions = None
                if st.session_state.auto:
                    st.session_state["_do_research"] = True
            else:
                st.session_state.questions = res.get("questions", [])
            st.rerun()
elif st.session_state.req:
    st.success("Requirements ready")
    with st.expander("Requirements", expanded=False):
        st.json(st.session_state.req)
else:
    st.caption("Run the pipeline (or Analyze alone) to populate requirements.")
    if st.button("Analyze only", key="analyze_only_btn"):
        st.session_state.idea = idea  # manual path still feeds research/extension
        try:
            res = rd_analyze(api_url, api_token, idea)
        except APIError as exc:
            st.error(f"Analyze failed: {exc}")
        else:
            if res.get("ready"):
                st.session_state.req = res["requirements"]
                st.session_state.questions = None
            else:
                st.session_state.questions = res.get("questions", [])
            st.rerun()


# ----------------------------------------------------------------------------
# 3 · Research — R&D orchestrator stream; on artifact, hand off to code when
# auto (one-shot `_do_code` flag, consumed by section 4 in this same run).
# ----------------------------------------------------------------------------
st.markdown("### 3 · Research")
if st.session_state.req is None:
    st.caption("Requirements needed first — run the pipeline above (or Analyze only).")
else:
    research_chain = st.session_state.pop("_do_research", False)
    research_btn = st.button("Start research", type="primary", key="research_btn")
    if research_chain or research_btn:
        try:
            resp = rd_research_stream(
                api_url, api_token, st.session_state.req, idea=st.session_state.idea
            )
        except APIError as exc:
            st.error(f"Research failed: {exc}")
        else:
            got_artifact = False
            try:
                with resp:
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        etype = evt.get("type")
                        if etype == "orchestrator":
                            st.markdown(
                                f"**Plan:** {len(evt.get('directions', []))} directions"
                            )
                            for d in evt.get("directions", []):
                                st.caption(f"• **{d['id']}**: {d['question'][:120]}")
                        elif etype == "subagent":
                            if evt["event"] == "start":
                                st.caption(
                                    f"🔎 **{evt['direction_id']}** — {evt['question'][:100]}"
                                )
                            else:
                                s = evt["summary"]
                                st.markdown(
                                    f"**{s['direction_id']}** ({s['confidence']}) — "
                                    f"{s['findings'][:300]}"
                                )
                        elif etype == "validator":
                            v = evt["validation"]
                            st.markdown(
                                f"**Validation** — feasible: {v['feasible']} · "
                                f"missing: {v.get('missing')}"
                            )
                            if v.get("risks"):
                                st.warning(f"Risks: {v['risks']}")
                        elif etype == "artifact":
                            st.session_state.artifact = evt["data"]
                            st.session_state.rd_run_id = evt.get("run_id")
                            got_artifact = True
                            st.success("Artifact ready")
                            with st.expander("Demo spec", expanded=False):
                                st.json(evt["data"].get("demo_spec", {}))
                            with st.expander("Full artifact"):
                                st.json(evt["data"])
                        elif etype == "error":
                            st.error(f"Backend error: {evt.get('delta')}")
            except Exception as exc:
                st.error(f"Stream interrupted: {exc}")
            if got_artifact and st.session_state.auto:
                # Continue the chain in section 4 of this same script run.
                st.session_state["_do_code"] = True


# ----------------------------------------------------------------------------
# 4 · Code — demo PoC from the artifact and/or extend against a codebase.
# ----------------------------------------------------------------------------
st.markdown("### 4 · Code")
if st.session_state.artifact is None:
    st.caption("Research artifact needed — finish research above to unlock the code stage.")
else:
    # One-shot chain flag from section 3; popped unconditionally so a stale flag
    # can never re-run a stream on a later widget-triggered rerun.
    chain_go = st.session_state.pop("_do_code", False)

    # -- 4a. Demo PoC (from the research artifact) ---------------------------
    if st.session_state.mode in ("demo", "both"):
        st.markdown("#### Demo PoC")
        demo_btn = st.button("Generate demo", key="demo_btn")
        if chain_go or demo_btn:
            st.session_state.demo = {
                "plan": None,
                "files": [],
                "tester": None,
                "sandbox": None,
                "artifact": None,
            }
            try:
                resp = code_generate_stream(
                    api_url, api_token, st.session_state.artifact
                )
            except APIError as exc:
                st.error(f"Code generation failed: {exc}")
            else:
                try:
                    with resp:
                        for line in resp.iter_lines(decode_unicode=True):
                            if not line:
                                continue
                            try:
                                evt = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            etype = evt.get("type")
                            event = evt.get("event")
                            if etype == "code" and event == "plan":
                                tasks = evt.get("file_tasks", [])
                                st.session_state.demo["plan"] = tasks
                                st.write(f"Plan: {len(tasks)} files")
                            elif etype == "code" and event == "file_start":
                                st.caption(
                                    f"🔨 {evt.get('path')} — {evt.get('goal', '')[:80]}"
                                )
                            elif etype == "code" and event == "file_result":
                                f = evt.get("file", {})
                                st.session_state.demo["files"].append(f)
                                lang = (
                                    "python"
                                    if f.get("path", "").endswith(".py")
                                    else "text"
                                )
                                st.code(f.get("content", "")[:2000], language=lang)
                            elif etype == "tester":
                                st.session_state.demo["tester"] = evt.get("file", {})
                                st.write("**Tester**")
                                st.code(
                                    evt.get("file", {}).get("content", "")[:2000],
                                    language="python",
                                )
                            elif etype == "sandbox" and event == "result":
                                s = evt.get("sandbox", {})
                                st.session_state.demo["sandbox"] = s
                                if s.get("passed"):
                                    st.success("Sandbox passed")
                                else:
                                    st.error("Sandbox failed")
                                    with st.expander("Sandbox logs"):
                                        st.code(s.get("pytest_log", "")[:2000])
                                        st.code(s.get("demo_log", "")[:2000])
                            elif etype == "code" and event == "artifact":
                                st.session_state.demo["artifact"] = evt.get("data")
                                st.success(
                                    f"Demo artifact ready — run {evt.get('run_id', '')[:8]}"
                                )
                                st.json(evt.get("data", {}).get("demo_spec", {}))
                                with st.expander("Full CodeArtifact"):
                                    st.json(evt.get("data"))
                            elif etype == "error":
                                st.error(f"Backend error: {evt.get('delta')}")
                except Exception as exc:
                    st.error(f"Stream interrupted: {exc}")

    # -- 4b. Extend a codebase (PoC patch, optional feasibility pre-check) ----
    if st.session_state.mode in ("extend", "both"):
        st.markdown("#### Extend a codebase")
        with st.expander("Repo targeting", expanded=not st.session_state.repo_id):
            ext_path = st.text_input(
                "Local repo path",
                placeholder="/abs/path/to/repo",
                key="ext_repo_path",
            )
            if st.button(
                "Index repo", key="ext_repo_index_btn", disabled=not ext_path.strip()
            ):
                try:
                    stats = repo_index(api_url, api_token, ext_path.strip())
                except APIError as exc:
                    st.error(f"Index failed: {exc}")
                else:
                    st.session_state.repo_id = stats["repo_id"]
                    st.success(
                        f"Indexed {stats['files']} files → {stats['chunks']} chunks "
                        f"({stats['skipped']} skipped) in {stats['runtime_ms']} ms"
                    )
            try:
                repos = repo_list(api_url, api_token)
            except APIError:
                repos = []
            if repos:
                options = {
                    f"{r.get('name')} · {r['id']} · {r.get('chunks', 0)} chunks": r["id"]
                    for r in repos
                }
                ids = list(options.values())
                current = st.session_state.repo_id
                picked = st.selectbox(
                    "Indexed repos",
                    list(options),
                    index=ids.index(current) if current in ids else 0,
                    key="ext_repo_pick",
                )
                st.session_state.repo_id = options[picked]
            st.caption(f"Target repo: {st.session_state.repo_id or '— none selected'}")

        # Feasibility pre-check (optional): run before patching, chain when auto.
        if st.session_state.feas_pre and st.session_state.repo_id:
            feas_chain = st.session_state.pop("_do_feas", False)
            feas_btn = st.button("Run feasibility", key="feas_btn")
            feas_go = feas_chain or feas_btn
            if chain_go and st.session_state.feas is None:
                feas_go = True
            if feas_go:
                try:
                    resp = feasibility_stream(
                        api_url,
                        api_token,
                        st.session_state.idea,
                        st.session_state.repo_id,
                        None,
                    )
                except APIError as exc:
                    st.error(f"Feasibility failed: {exc}")
                else:
                    feas_ok = False
                    try:
                        with resp:
                            for line in resp.iter_lines(decode_unicode=True):
                                if not line:
                                    continue
                                try:
                                    evt = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                if evt.get("type") != "feasibility":
                                    continue
                                event = evt.get("event")
                                if event == "start":
                                    st.caption(
                                        f"run {evt.get('run_id', '')[:8]} · "
                                        f"repo {evt.get('repo_id')}"
                                    )
                                elif event == "clarify":
                                    st.warning(
                                        f"Clarification needed: {evt.get('questions')}"
                                    )
                                elif event == "understand":
                                    with st.expander("Understanding", expanded=True):
                                        st.json(evt["understanding"])
                                elif event == "map":
                                    cmap = evt["codebase_map"]
                                    st.markdown(
                                        f"**Map:** {len(cmap.get('modules', []))} modules · "
                                        f"{len(cmap.get('dependency_edges', []))} edges · "
                                        f"{len(cmap.get('impacted_modules', []))} impacted"
                                    )
                                    if cmap.get("extension_points"):
                                        st.caption(
                                            "extension points: "
                                            + ", ".join(
                                                f"{ep['path']}::{ep.get('symbol', '')}"
                                                for ep in cmap["extension_points"]
                                            )
                                        )
                                    if cmap.get("data_missing"):
                                        st.caption(f"data missing: {cmap['data_missing']}")
                                elif event == "claim":
                                    c = evt["claim"]
                                    refs = ", ".join(
                                        f"{r['path']}:{r.get('start_line')}-{r.get('end_line')}"
                                        for r in c.get("evidence", [])
                                        if r.get("valid")
                                    )
                                    st.markdown(
                                        f"**[{c['dimension']}/{c['status']}]** "
                                        f"{c['statement'][:200]}"
                                    )
                                    if refs:
                                        st.caption(f"evidence: {refs}")
                                    elif c.get("evidence"):
                                        st.caption("no valid evidence")
                                elif event == "report":
                                    r = evt["data"]
                                    st.session_state.feas = r
                                    feas_ok = True
                                    if r["verdict"] == "plausible":
                                        st.success(f"Verdict: {r['verdict']}")
                                    elif r["verdict"] == "plausible_with_risks":
                                        st.warning(f"Verdict: {r['verdict']}")
                                    else:
                                        st.error(f"Verdict: {r['verdict']}")
                                    if r["verdict"] in (
                                        "plausible",
                                        "plausible_with_risks",
                                    ):
                                        st.info(
                                            "Continue to extend — the PoC patch proceeds."
                                        )
                                    else:
                                        st.warning(
                                            "Proceeding to extend anyway — results may "
                                            "be unreliable."
                                        )
                                    if r.get("blockers"):
                                        st.warning("Blockers: " + "; ".join(r["blockers"]))
                                    if r.get("alternatives"):
                                        st.info(
                                            "Alternatives: " + " | ".join(r["alternatives"])
                                        )
                                    if r.get("reframed_idea"):
                                        st.markdown(
                                            f"**Reframed idea:** {r['reframed_idea']}"
                                        )
                                    with st.expander("Full report"):
                                        st.json(r)
                    except Exception as exc:
                        st.error(f"Stream interrupted: {exc}")
                    if feas_ok and chain_go:
                        st.session_state["_do_extend"] = True

        ext_chain = st.session_state.pop("_do_extend", False)
        ext_btn = st.button(
            "Run extend",
            type="primary",
            key="ext_btn",
            disabled=not st.session_state.repo_id,
        )
        ext_go = ext_chain or ext_btn
        if (
            chain_go
            and (not st.session_state.feas_pre or st.session_state.feas is not None)
            and st.session_state.repo_id
        ):
            ext_go = True
        if ext_go:
            st.session_state.ext = {
                "context": None,
                "plan": None,
                "changes": [],
                "tester": None,
                "sandbox": None,
                "artifact": None,
            }
            try:
                resp = code_extend_stream(
                    api_url,
                    api_token,
                    st.session_state.idea,
                    repo_id=st.session_state.repo_id or None,
                    path=None,
                    entrypoint=None,
                )
            except APIError as exc:
                st.error(f"Extend failed: {exc}")
            else:
                try:
                    with resp:
                        for line in resp.iter_lines(decode_unicode=True):
                            if not line:
                                continue
                            try:
                                evt = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            etype = evt.get("type")
                            if etype == "extend_start":
                                st.caption(f"run {evt.get('run_id', '')[:8]}")
                            elif etype == "context":
                                st.session_state.ext["context"] = evt
                                st.info(
                                    f"repo mode {evt.get('repo_mode')} — "
                                    f"{evt.get('n_files', 0)} files, "
                                    f"{len(evt.get('hits', []))} hits"
                                )
                            elif etype == "plan":
                                tasks = evt.get("file_tasks", [])
                                st.session_state.ext["plan"] = tasks
                                st.write(f"Plan: {len(tasks)} changes")
                                for t in tasks:
                                    badge = (
                                        "✏️ modify"
                                        if t.get("action") == "modify"
                                        else "✨ create"
                                    )
                                    st.write(f"{badge} — {t['path']}")
                            elif etype == "change_start":
                                st.caption(f"⏳ {evt.get('action')} {evt.get('path')}")
                            elif etype == "change_result":
                                f = evt.get("file", {})
                                st.session_state.ext["changes"].append(f)
                                st.write(f"**{f.get('path')}** ({f.get('action')})")
                                st.code(
                                    f.get("content", "") or "(search/replace blocks)"
                                )
                            elif etype == "tester":
                                st.session_state.ext["tester"] = evt.get("file", {})
                                st.write("🧪 **Tester**")
                                with st.expander("Test file"):
                                    st.code(
                                        evt.get("file", {}).get("content", ""),
                                        language="python",
                                    )
                            elif etype == "sandbox" and evt.get("event") == "result":
                                sb = evt.get("sandbox", {})
                                st.session_state.ext["sandbox"] = sb
                                if sb.get("passed"):
                                    st.success("✅ repo tests passed")
                                else:
                                    st.error("❌ sandbox failed")
                                with st.expander("Sandbox logs"):
                                    st.json(sb)
                            elif etype == "artifact":
                                st.session_state.ext["artifact"] = evt.get("data")
                                st.success("Extend artifact ready")
                                with st.expander("Full extend artifact"):
                                    st.json(evt.get("data"))
                            elif etype == "extend_error":
                                st.error(evt.get("message", "unknown error"))
                                break
                except Exception as exc:
                    st.error(f"Stream interrupted: {exc}")

    # -- Downloads: everything accumulated in session state, re-offered here --
    demo_run = st.session_state.demo or {}
    for i, f in enumerate(demo_run.get("files") or []):
        content = (f or {}).get("content") or ""
        if content:
            name = os.path.basename(f.get("path") or "demo_file")
            st.download_button(
                f"⬇ {name}", data=content, file_name=name, key=f"dl_demo_{i}"
            )
    demo_tester = demo_run.get("tester") or {}
    if demo_tester.get("content"):
        name = os.path.basename(demo_tester.get("path") or "test_demo.py")
        st.download_button(
            f"⬇ {name}",
            data=demo_tester["content"],
            file_name=name,
            key="dl_demo_tester",
        )
    ext_run = st.session_state.ext or {}
    for i, f in enumerate(ext_run.get("changes") or []):
        content = (f or {}).get("content") or ""
        if content:
            name = os.path.basename(f.get("path") or "ext_file")
            st.download_button(
                f"⬇ {name}", data=content, file_name=name, key=f"dl_ext_{i}"
            )
    ext_tester = ext_run.get("tester") or {}
    if ext_tester.get("content"):
        name = os.path.basename(ext_tester.get("path") or "test_ext.py")
        st.download_button(
            f"⬇ {name}",
            data=ext_tester["content"],
            file_name=name,
            key="dl_ext_tester",
        )


# ----------------------------------------------------------------------------
# 5 · Result — compact summary of the run (renders after streams finish).
# ----------------------------------------------------------------------------
st.markdown("### 5 · Result")
if st.session_state.demo or st.session_state.ext:
    research_status = "—"
    if st.session_state.artifact:
        run_id = (st.session_state.rd_run_id or "")[:8]
        research_status = f"✅ artifact ready (run {run_id})"
    summary = [
        ("Research", research_status),
        ("Demo sandbox", _sandbox_status(st.session_state.demo)),
        ("Extend sandbox", _sandbox_status(st.session_state.ext)),
    ]
    if st.session_state.feas:
        summary.append(("Feasibility", str(st.session_state.feas.get("verdict", "—"))))
    for label, value in summary:
        st.markdown(f"- **{label}:** {value}")

    demo_passed = bool(((st.session_state.demo or {}).get("sandbox") or {}).get("passed"))
    ext_passed = bool(((st.session_state.ext or {}).get("sandbox") or {}).get("passed"))
    if demo_passed or ext_passed:
        st.success("Pipeline finished — at least one sandbox passed.")
    else:
        st.warning("Pipeline finished — no sandbox passed (or none ran yet).")

    if st.session_state.demo and st.session_state.demo.get("artifact"):
        with st.expander("Demo artifact"):
            st.json(st.session_state.demo["artifact"])
    if st.session_state.ext and st.session_state.ext.get("artifact"):
        with st.expander("Extend artifact"):
            st.json(st.session_state.ext["artifact"])
else:
    st.caption("No results yet — run the pipeline.")


# ----------------------------------------------------------------------------
# Utilities — old tab bodies condensed into nested expanders (documents, ask,
# memory, runs). These are standalone tools and never touch the run state.
# ----------------------------------------------------------------------------
st.divider()
with st.expander("🧰 Utilities — documents · ask · memory · runs", expanded=False):
    st.caption("Standalone backend tools; they do not affect the run above.")

    # -- Documents: upload + server list + per-row actions + session status --
    with st.expander("📄 Documents — upload · delete · re-ingest"):
        st.markdown("**Upload a PDF**")
        uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
        title = st.text_input("Title (metadata)", placeholder="optional")
        if st.button("Upload", disabled=uploaded is None):
            try:
                meta = {"title": title} if title.strip() else {}
                doc = upload_document(
                    api_url, api_token, uploaded.name, uploaded.getvalue(), meta
                )
            except APIError as exc:
                st.error(f"Upload failed: {exc}")
            else:
                st.session_state.setdefault("docs", [])
                st.session_state.docs.append(
                    {"id": doc["document_id"], "name": uploaded.name}
                )
                st.success(f"Queued as {doc['document_id']}")

        st.markdown("**All documents (server)**")
        if st.button("Refresh server list"):
            try:
                server_docs = list_documents(api_url, api_token)
                st.session_state["server_docs"] = server_docs
            except APIError as exc:
                st.error(str(exc))
        server_docs = st.session_state.get("server_docs")
        if server_docs is None:
            try:
                server_docs = list_documents(api_url, api_token)
                st.session_state["server_docs"] = server_docs
            except Exception:
                # Backend may be down at first render — keep the page alive.
                server_docs = []
        if server_docs:
            df = pd.DataFrame(
                [
                    {
                        "id": d.get("id"),
                        "title": d.get("title"),
                        "status": d.get("status"),
                        "chunks": d.get("text_chunks"),
                        "images": d.get("images"),
                        "error": (d.get("error") or "")[:80],
                    }
                    for d in server_docs
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            sel = st.selectbox(
                "Select doc for actions",
                options=[d.get("id") for d in server_docs] or [],
                key="doc_sel",
            )
            c1, c2 = st.columns(2)
            if c1.button("Delete selected"):
                try:
                    document_delete(api_url, api_token, sel)
                    st.success(f"Deleted {sel}")
                    st.session_state["server_docs"] = list_documents(
                        api_url, api_token
                    )
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))
            if c2.button("Re-ingest selected"):
                try:
                    res = document_reingest(api_url, api_token, sel)
                    st.json(res)
                    st.success(f"Re-ingested {sel}")
                except APIError as exc:
                    st.error(str(exc))
        else:
            st.caption("No documents on server.")

        st.markdown("**Ingestion status (session)**")
        docs = st.session_state.get("docs", [])
        if docs:
            rows = []
            for d in docs:
                try:
                    info = document_status(api_url, api_token, d["id"])
                except APIError as exc:
                    rows.append(
                        {
                            "document_id": d["id"],
                            "name": d["name"],
                            "status": f"error: {exc}",
                        }
                    )
                    continue
                rows.append(
                    {
                        "document_id": d["id"],
                        "name": d["name"],
                        "status": info.get("status"),
                        "text_chunks": info.get("text_chunks", "-"),
                        "images": info.get("images", "-"),
                        "error": (info.get("error") or "")[:120],
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if st.button("Refresh status", type="secondary"):
                st.rerun()
        else:
            st.caption("No documents uploaded yet in this session.")
        st.caption("Ingestion runs asynchronously — refresh to see progress.")

    # -- Ask: retrieve + streamed answer over ingested documents ------------
    with st.expander("🔎 Ask — retrieval + streamed answer"):
        question = st.text_area(
            "Question", placeholder="e.g. What is the main contribution of this paper?"
        )
        col_k, col_ki = st.columns(2)
        k = col_k.slider("Text chunks", 1, 10, 4)
        k_images = col_ki.slider("Figures", 0, 10, 2)

        if st.button("Ask", type="primary", disabled=not question.strip()):
            try:
                resp = query_stream(api_url, api_token, question, k, k_images)
            except APIError as exc:
                st.error(f"Query failed: {exc}")
            else:
                with resp:
                    lines = resp.iter_lines(decode_unicode=True)
                    try:
                        first = json.loads(next(lines))
                    except StopIteration:
                        st.warning("No response from server.")
                        first = None

                    if first:
                        items = first.get("items", [])
                        st.write(f"**Sources** ({len(items)})")
                        for it in items:
                            if it["kind"] == "image":
                                try:
                                    img = get_file(
                                        api_url,
                                        api_token,
                                        it["image_url"].removeprefix("/files/"),
                                    )
                                    st.image(
                                        img,
                                        caption=f"page {it['page']} · score {it['score']}",
                                        width=320,
                                    )
                                except APIError as exc:
                                    st.warning(f"Could not load figure: {exc}")
                            else:
                                st.markdown(
                                    f"**[page {it['page']}]** {it['content']} — "
                                    f"*score {it['score']}*"
                                )
                        st.divider()

                        st.write("**Answer**")

                        def deltas():
                            try:
                                for line in lines:
                                    if not line:
                                        continue
                                    try:
                                        evt = json.loads(line)
                                    except json.JSONDecodeError:
                                        continue
                                    if evt.get("type") == "text":
                                        yield evt.get("delta", "")
                                    elif evt.get("type") == "error":
                                        yield f"\n\n[error] {evt.get('delta','unknown')}"
                            except Exception as exc:
                                # ChunkedEncodingError, ConnectionError, etc.
                                yield f"\n\n[stream interrupted: {exc}]"

                        try:
                            st.write_stream(deltas())
                        except Exception as exc:
                            st.error(f"Stream failed: {exc}")
                            st.caption(
                                "Tip: check `docker compose -f deploy/docker-compose.yml "
                                "logs api` for backend errors; retry with smaller k or "
                                "check PRODRAG_CHAT_API_KEY."
                            )

    # -- Memory: episodic search + episodes + conversations -----------------
    with st.expander("🧠 Memory — episodes · conversations"):
        q = st.text_input("Search episodic memory", placeholder="e.g. attention mechanism")
        k = st.slider("Top k", 1, 20, 5, key="mem_k")
        if st.button("Search memory", disabled=not q.strip()):
            try:
                res = memory_search(api_url, api_token, q, k)
                st.json(res)
                for r in res.get("results", []):
                    st.markdown(
                        f"**Q:** {r.get('question')} — **A:** {r.get('answer','')[:300]} "
                        f"— *score {r.get('score'):.2f}*"
                    )
            except APIError as exc:
                st.error(str(exc))
        if st.button("List episodes"):
            try:
                eps = memory_episodes(api_url, api_token, limit=20)
                st.dataframe(pd.DataFrame(eps), use_container_width=True, hide_index=True)
            except APIError as exc:
                st.error(str(exc))

        st.markdown("**Conversations**")
        if st.button("List conversations"):
            try:
                convs = list_conversations(api_url, api_token, limit=20)
                st.dataframe(
                    pd.DataFrame(convs), use_container_width=True, hide_index=True
                )
                sel = st.selectbox(
                    "Select conversation",
                    options=[c.get("id") for c in convs] or [],
                    key="conv_sel",
                )
                if sel and st.button("Show messages"):
                    msgs = get_conversation_messages(api_url, api_token, sel, n=20)
                    st.json(msgs)
            except APIError as exc:
                st.error(str(exc))

    # -- Runs: research + code history --------------------------------------
    with st.expander("🗂 Runs — research · code history"):
        st.caption("TTL: research 30d / code 30d / episodes 7d / conv 1d (lazy expiry on list)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Research runs**")
            if st.button("Refresh research runs"):
                try:
                    runs = list_rd_runs(api_url, api_token, limit=20)
                    st.session_state["rd_runs"] = runs
                except APIError as exc:
                    st.error(str(exc))
            runs = st.session_state.get("rd_runs")
            if runs is None:
                try:
                    runs = list_rd_runs(api_url, api_token, limit=20)
                    st.session_state["rd_runs"] = runs
                except Exception:
                    # Backend may be down at first render — keep the page alive.
                    runs = []
            if runs:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "id": r.get("id")[:8],
                                "idea": (r.get("idea") or "")[:60],
                                "status": r.get("status"),
                                "created_at": r.get("created_at"),
                            }
                            for r in runs
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                sel = st.selectbox(
                    "Select research run",
                    options=[r.get("id") for r in runs] or [],
                    key="rd_run_sel",
                )
                if sel and st.button("Show research artifact"):
                    try:
                        det = get_rd_run(api_url, api_token, sel)
                        st.json(det)
                    except APIError as exc:
                        st.error(str(exc))
            else:
                st.caption("No research runs yet.")
        with c2:
            st.markdown("**Code runs**")
            if st.button("Refresh code runs"):
                try:
                    cruns = list_code_runs(api_url, api_token, limit=20)
                    st.session_state["code_runs"] = cruns
                except APIError as exc:
                    st.error(str(exc))
            cruns = st.session_state.get("code_runs")
            if cruns is None:
                try:
                    cruns = list_code_runs(api_url, api_token, limit=20)
                    st.session_state["code_runs"] = cruns
                except Exception:
                    # Backend may be down at first render — keep the page alive.
                    cruns = []
            if cruns:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "id": r.get("id")[:8],
                                "status": r.get("status"),
                                "created_at": r.get("created_at"),
                            }
                            for r in cruns
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                sel = st.selectbox(
                    "Select code run",
                    options=[r.get("id") for r in cruns] or [],
                    key="code_run_sel",
                )
                if sel and st.button("Show code artifact"):
                    try:
                        det = get_code_run(api_url, api_token, sel)
                        st.json(det)
                    except APIError as exc:
                        st.error(str(exc))
            else:
                st.caption("No code runs yet.")
