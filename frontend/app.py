"""ProdRag Streamlit UI: upload PDFs, track ingestion, ask questions.

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
    document_status,
    get_file,
    health,
    query_stream,
    rd_analyze,
    rd_research_stream,
    research_stream,
    upload_document,
)

st.set_page_config(page_title="ProdRag", page_icon="📚", layout="wide")


def _default(name: str, fallback: str) -> str:
    try:
        return st.secrets.get(name, os.environ.get(name, fallback))
    except Exception:
        return os.environ.get(name, fallback)


# ----------------------------------------------------------------------------
# Sidebar: connection settings
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

tab_upload, tab_query, tab_research, tab_rd = st.tabs(["Documents", "Ask", "Research", "R&D"])


# ----------------------------------------------------------------------------
# Documents tab: upload + async ingestion status
# ----------------------------------------------------------------------------
with tab_upload:
    st.subheader("Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    title = st.text_input("Title (metadata)", placeholder="optional")
    if st.button("Upload", disabled=uploaded is None):
        try:
            meta = {"title": title} if title.strip() else {}
            doc = upload_document(api_url, api_token, uploaded.name, uploaded.getvalue(), meta)
        except APIError as exc:
            st.error(f"Upload failed: {exc}")
        else:
            st.session_state.setdefault("docs", [])
            st.session_state.docs.append(
                {"id": doc["document_id"], "name": uploaded.name}
            )
            st.success(f"Queued as {doc['document_id']}")

    st.divider()
    st.subheader("Ingestion status")
    docs = st.session_state.get("docs", [])
    if docs:
        rows = []
        for d in docs:
            try:
                info = document_status(api_url, api_token, d["id"])
            except APIError as exc:
                rows.append({"document_id": d["id"], "name": d["name"], "status": f"error: {exc}"})
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
        st.caption("No documents uploaded yet.")
    st.caption("Ingestion runs asynchronously — refresh to see progress.")


# ----------------------------------------------------------------------------
# Ask tab: retrieve + streamed answer
# ----------------------------------------------------------------------------
with tab_query:
    st.subheader("Ask your documents")
    question = st.text_area("Question", placeholder="e.g. What is the main contribution of this paper?")
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
                                    api_url, api_token, it["image_url"].removeprefix("/files/")
                                )
                                st.image(img, caption=f"page {it['page']} · score {it['score']}", width=320)
                            except APIError as exc:
                                st.warning(f"Could not load figure: {exc}")
                        else:
                            st.markdown(
                                f"**[page {it['page']}]** {it['content']} — *score {it['score']}*"
                            )
                    st.divider()

                    st.write("**Answer**")

                    def deltas():
                        for line in lines:
                            if not line:
                                continue
                            evt = json.loads(line)
                            if evt.get("type") == "text":
                                yield evt.get("delta", "")

                    st.write_stream(deltas())


# ----------------------------------------------------------------------------
# Research tab: agentic research with conversation + episodic memory
# ----------------------------------------------------------------------------
with tab_research:
    st.subheader("Research with the agent")
    question = st.text_area(
        "Research question", placeholder="e.g. Compare how these papers handle causal reasoning in videos"
    )
    col_rk, col_rki = st.columns(2)
    rk = col_rk.slider("KB text chunks", 1, 10, 4)
    rk_images = col_rki.slider("KB figures", 0, 5, 1)

    if st.button("Research", type="primary", disabled=not question.strip()):
        sid = st.session_state.get("research_session")
        try:
            resp = research_stream(api_url, api_token, question, sid, rk, rk_images)
        except APIError as exc:
            st.error(f"Research failed: {exc}")
        else:
            with resp:
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    evt = json.loads(line)
                    etype = evt.get("type")
                    if etype == "memory" and evt.get("event") == "session":
                        st.session_state["research_session"] = evt["session_id"]
                        st.info(f"Session {evt['session_id'][:8]}…")
                    elif etype == "memory":
                        st.success(f"💾 remembered in session {evt['session_id'][:8]}")
                    elif etype == "agent":
                        if evt["event"] == "tool_call":
                            st.markdown(f"🔎 **{evt['name']}** `{json.dumps(evt['args'])[:120]}`")
                        else:
                            st.markdown(f"  ↳ {evt['summary']}")
                    elif etype == "sources":
                        for it in evt["items"]:
                            if it["kind"] == "image":
                                try:
                                    img = get_file(
                                        api_url, api_token, it["image_url"].removeprefix("/files/")
                                    )
                                    st.image(img, caption=f"page {it['page']} · {it['score']}", width=300)
                                except APIError:
                                    pass
                            else:
                                st.markdown(f"**[page {it['page']}]** {it['content']} — *{it['score']}*")
                    elif etype == "text":
                        st.markdown(evt.get("delta", ""))


# ----------------------------------------------------------------------------
# R&D tab: idea -> research -> artifact (runnable demo spec)
# ----------------------------------------------------------------------------
with tab_rd:
    st.subheader("R&D — idea to demo")
    st.caption("Analyzer asks 1–3 clarifying questions if needed (second-request handshake). Then 5 parallel subagents research fixed directions.")

    idea = st.text_area("Idea / paper", placeholder="e.g. Turn the 'Attention Is All You Need' core idea into a minimal demo on toy data", key="rd_idea")
    if "rd_requirements" not in st.session_state:
        st.session_state["rd_requirements"] = None
    if "rd_questions" not in st.session_state:
        st.session_state["rd_questions"] = None

    col_analyze, col_clear = st.columns([1, 1])
    if col_analyze.button("Analyze", type="primary", disabled=not idea.strip()):
        try:
            res = rd_analyze(api_url, api_token, idea, st.session_state.get("research_session"))
        except APIError as exc:
            st.error(f"Analyze failed: {exc}")
        else:
            if res.get("ready"):
                st.session_state["rd_requirements"] = res["requirements"]
                st.session_state["rd_questions"] = None
                st.success("Requirements ready")
            else:
                st.session_state["rd_requirements"] = None
                st.session_state["rd_questions"] = res.get("questions", [])
                st.warning(f"Need clarification: {res.get('questions')}")

    if st.session_state.get("rd_questions"):
        st.info("Answer the clarifying questions, then re-analyze.")
        answers = []
        for i, q in enumerate(st.session_state["rd_questions"]):
            ans = st.text_input(f"Q{i+1}: {q}", key=f"rd_ans_{i}")
            answers.append(ans)
        if st.button("Submit answers"):
            try:
                res = rd_analyze(api_url, api_token, idea, st.session_state.get("research_session"), [a for a in answers if a.strip()])
            except APIError as exc:
                st.error(f"Analyze failed: {exc}")
            else:
                if res.get("ready"):
                    st.session_state["rd_requirements"] = res["requirements"]
                    st.session_state["rd_questions"] = None
                    st.success("Requirements ready")
                    st.rerun()
                else:
                    st.session_state["rd_questions"] = res.get("questions", [])

    req = st.session_state.get("rd_requirements")
    if req:
        with st.expander("Requirements", expanded=True):
            st.json(req)
        col_rk, col_rki = st.columns(2)
        rk = col_rk.slider("k per subagent", 1, 10, 4, key="rd_k")
        rk_images = col_rki.slider("Figures per subagent", 0, 5, 1, key="rd_ki")

        if st.button("Start research", type="primary", key="rd_start"):
            try:
                resp = rd_research_stream(api_url, api_token, req, st.session_state.get("research_session"), idea, rk, rk_images)
            except APIError as exc:
                st.error(f"R&D research failed: {exc}")
            else:
                with resp:
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        evt = json.loads(line)
                        etype = evt.get("type")
                        if etype == "orchestrator":
                            st.markdown(f"**Plan:** {len(evt.get('directions', []))} directions")
                            for d in evt.get("directions", []):
                                st.caption(f"• **{d['id']}**: {d['question'][:120]}")
                        elif etype == "subagent":
                            if evt["event"] == "start":
                                st.markdown(f"🔎 **{evt['direction_id']}** — {evt['question'][:100]}")
                            else:
                                s = evt["summary"]
                                st.markdown(f"**{s['direction_id']}** ({s['confidence']}) — {s['findings'][:300]}")
                                if s.get("gaps"):
                                    st.caption(f"gaps: {s['gaps']}")
                                for it in s.get("sources", [])[:2]:
                                    if it["kind"] == "image":
                                        try:
                                            img = get_file(api_url, api_token, it["image_url"].removeprefix("/files/"))
                                            st.image(img, caption=f"page {it['page']} · {it['score']}", width=300)
                                        except APIError:
                                            pass
                                    else:
                                        st.markdown(f"[page {it['page']}] {it['content'][:200]} — *{it['score']}*")
                        elif etype == "validator":
                            v = evt["validation"]
                            st.divider()
                            st.markdown(f"**Validation** — feasible: {v['feasible']} · missing: {v.get('missing')}")
                            if v.get("risks"):
                                st.warning(f"Risks: {v['risks']}")
                        elif etype == "artifact":
                            st.success("Artifact ready — handoff to coding module")
                            st.json(evt["data"]["demo_spec"])
                            with st.expander("Full artifact"):
                                st.json(evt["data"])
                            st.caption(f"Sources: {len(evt['data'].get('sources', []))} · run {evt.get('run_id','')[:8]}")

    if col_clear.button("Clear R&D state"):
        st.session_state["rd_requirements"] = None
        st.session_state["rd_questions"] = None
        st.rerun()
