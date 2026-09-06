"""ProdRag Streamlit UI: upload PDFs, track ingestion, ask questions.

Run:
    uv sync --extra frontend
    cd frontend && uv run streamlit run app.py
"""

import json
import os
import time

import pandas as pd
import streamlit as st

from api import (
    APIError,
    code_analyze,
    code_generate_stream,
    document_delete,
    document_reingest,
    document_status,
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
    research_stream,
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
    # TTL + search status (v2.0)
    try:
        # Use cached search status if available, else probe
        if st.button("Search status"):
            res = web_search_probe(api_url, api_token, "test", k=1)
            st.json(res)
    except Exception:
        pass
    st.caption("TTL: research 30d / code 30d / episodes 7d / conv 1d (lazy)")

tab_upload, tab_query, tab_research, tab_rd, tab_code, tab_memory, tab_runs = st.tabs(
    ["Documents", "Ask", "Research", "R&D", "Code", "Memory", "Runs"]
)


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
    # v2.0: list from server on load + per-row actions
    st.subheader("All documents (server)")
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
        except APIError:
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
        # per-row actions
        sel = st.selectbox("Select doc for actions", options=[d.get("id") for d in server_docs] or [], key="doc_sel")
        c1, c2 = st.columns(2)
        if c1.button("Delete selected"):
            try:
                document_delete(api_url, api_token, sel)
                st.success(f"Deleted {sel}")
                st.session_state["server_docs"] = list_documents(api_url, api_token)
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

    st.divider()
    st.subheader("Ingestion status (session)")
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
        st.caption("No documents uploaded yet in this session.")
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
                        except Exception as exc:  # ChunkedEncodingError, ConnectionError, etc.
                            yield f"\n\n[stream interrupted: {exc}]"

                    try:
                        st.write_stream(deltas())
                    except Exception as exc:
                        st.error(f"Stream failed: {exc}")
                        st.caption("Tip: check `docker compose -f deploy/docker-compose.yml logs api` for backend errors; retry with smaller k or check PRODRAG_CHAT_API_KEY.")


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
                                elif it["kind"] == "web":
                                    st.markdown(f"🌐 **[web]** [{it['source']}]({it['source']}) — {it['content'][:200]}")
                                else:
                                    st.markdown(f"**[page {it['page']}]** {it['content']} — *{it['score']}*")
                        elif etype == "text":
                            st.markdown(evt.get("delta", ""))
                        elif etype == "error":
                            st.error(f"Backend error: {evt.get('delta')}")
            except Exception as exc:
                st.error(f"Stream interrupted: {exc}")
                st.caption("Check `docker compose -f deploy/docker-compose.yml logs api` and retry.")


# ----------------------------------------------------------------------------
# R&D tab: idea -> research -> artifact (runnable demo spec)
# ----------------------------------------------------------------------------
with tab_rd:
    st.subheader("R&D — idea to demo")
    st.caption("Analyzer asks 1–3 clarifying questions if needed (second-request handshake). Then 5 subagents research fixed directions (web fallback when RAG<2).")

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
                                        elif it["kind"] == "web":
                                            st.markdown(f"🌐 [{it['source']}]({it['source']}) — {it['content'][:120]}")
                                        else:
                                            st.markdown(f"[page {it['page']}] {it['content'][:200]} — *{it['score']}*")
                            elif etype == "validator":
                                v = evt["validation"]
                                st.divider()
                                st.markdown(f"**Validation** — feasible: {v['feasible']} · missing: {v.get('missing')}")
                                if v.get("risks"):
                                    st.warning(f"Risks: {v['risks']}")
                            elif etype == "artifact":
                                st.session_state["last_artifact"] = evt["data"]
                                st.session_state["last_run_id"] = evt.get("run_id")
                                st.success("Artifact ready — handoff to Code tab")
                                st.json(evt["data"]["demo_spec"])
                                with st.expander("Full artifact"):
                                    st.json(evt["data"])
                                st.caption(f"Sources: {len(evt['data'].get('sources', []))} · run {evt.get('run_id','')[:8]}")
                            elif etype == "error":
                                st.error(f"Backend error: {evt.get('delta')}")
                except Exception as exc:
                    st.error(f"Stream interrupted: {exc}")
                    st.caption("Check `docker compose -f deploy/docker-compose.yml logs api` and retry.")

    if col_clear.button("Clear R&D state"):
        st.session_state["rd_requirements"] = None
        st.session_state["rd_questions"] = None
        st.rerun()


# ----------------------------------------------------------------------------
# Code tab: artifact -> demo.py + tests + sandbox
# ----------------------------------------------------------------------------
with tab_code:
    st.subheader("Code — artifact to demo")
    st.caption("Analyzer → orchestrator (fixed 3 files) → subagents (one LLM/file, web fallback when thin) → tester (from spec) → sandbox (tmpdir+subprocess).")

    artifact = st.session_state.get("last_artifact")
    if artifact is None:
        st.info("No artifact yet — run R&D above, or paste a ResearchArtifact JSON.")
        pasted = st.text_area("Paste artifact JSON", height=120, key="code_paste")
        if pasted.strip():
            try:
                artifact = json.loads(pasted)
                st.session_state["last_artifact"] = artifact
            except Exception as exc:
                st.error(f"Invalid JSON: {exc}")

    if st.session_state.get("last_artifact"):
        with st.expander("Artifact", expanded=False):
            st.json(st.session_state["last_artifact"])
        if st.button("Analyze artifact", type="primary"):
            try:
                res = code_analyze(api_url, api_token, st.session_state["last_artifact"])
                st.session_state["coding_spec"] = res.get("coding_spec", res)
                st.json(res)
                st.success("CodingSpec ready")
            except APIError as exc:
                st.error(str(exc))
        spec = st.session_state.get("coding_spec")
        if spec:
            with st.expander("CodingSpec", expanded=True):
                st.json(spec)
            if st.button("Generate demo (stream)"):
                try:
                    resp = code_generate_stream(api_url, api_token, st.session_state["last_artifact"])
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
                                if etype == "code" and evt.get("event") == "plan":
                                    st.write(f"Plan: {len(evt.get('file_tasks', []))} files")
                                elif etype == "code" and evt.get("event") == "file_start":
                                    st.caption(f"🔨 {evt.get('path')} — {evt.get('goal','')[:80]}")
                                elif etype == "code" and evt.get("event") == "file_result":
                                    f = evt.get("file", {})
                                    st.code(f.get("content","")[:2000], language="python" if f.get("path","").endswith(".py") else "text")
                                elif etype == "tester":
                                    st.write("**Tester**")
                                    st.code(evt.get("file", {}).get("content","")[:2000], language="python")
                                elif etype == "sandbox":
                                    if evt.get("event") == "result":
                                        s = evt.get("sandbox", {})
                                        st.json(s)
                                        if s.get("passed"):
                                            st.success("Sandbox passed")
                                        else:
                                            st.error("Sandbox failed")
                                            st.code(s.get("pytest_log","")[:2000])
                                            st.code(s.get("demo_log","")[:2000])
                                elif etype == "code" and evt.get("event") == "artifact":
                                    st.success(f"Code artifact ready — run {evt.get('run_id','')[:8]}")
                                    st.json(evt.get("data", {}).get("demo_spec", {}))
                                    st.session_state["last_code_artifact"] = evt.get("data")
                                    with st.expander("Full CodeArtifact"):
                                        st.json(evt.get("data"))
                                elif etype == "error":
                                    st.error(f"Backend error: {evt.get('delta')}")
                    except Exception as exc:
                        st.error(f"Stream interrupted: {exc}")
                        st.caption("Check `docker compose -f deploy/docker-compose.yml logs api` and retry.")
                except APIError as exc:
                    st.error(str(exc))


# ----------------------------------------------------------------------------
# Memory tab: episodic + conversations
# ----------------------------------------------------------------------------
with tab_memory:
    st.subheader("Memory")
    q = st.text_input("Search episodic memory", placeholder="e.g. attention mechanism")
    k = st.slider("Top k", 1, 20, 5, key="mem_k")
    if st.button("Search memory", disabled=not q.strip()):
        try:
            res = memory_search(api_url, api_token, q, k)
            st.json(res)
            for r in res.get("results", []):
                st.markdown(f"**Q:** {r.get('question')} — **A:** {r.get('answer','')[:300]} — *score {r.get('score'):.2f}*")
        except APIError as exc:
            st.error(str(exc))
    if st.button("List episodes"):
        try:
            eps = memory_episodes(api_url, api_token, limit=20)
            st.dataframe(pd.DataFrame(eps), use_container_width=True, hide_index=True)
        except APIError as exc:
            st.error(str(exc))
    st.divider()
    st.subheader("Conversations")
    if st.button("List conversations"):
        try:
            convs = list_conversations(api_url, api_token, limit=20)
            st.dataframe(pd.DataFrame(convs), use_container_width=True, hide_index=True)
            sel = st.selectbox("Select conversation", options=[c.get("id") for c in convs] or [], key="conv_sel")
            if sel and st.button("Show messages"):
                msgs = get_conversation_messages(api_url, api_token, sel, n=20)
                st.json(msgs)
        except APIError as exc:
            st.error(str(exc))


# ----------------------------------------------------------------------------
# Runs tab: history of research + code runs
# ----------------------------------------------------------------------------
with tab_runs:
    st.subheader("Runs History")
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
            except APIError:
                runs = []
        if runs:
            st.dataframe(pd.DataFrame([{"id": r.get("id")[:8], "idea": (r.get("idea") or "")[:60], "status": r.get("status"), "created_at": r.get("created_at")} for r in runs]), use_container_width=True, hide_index=True)
            sel = st.selectbox("Select research run", options=[r.get("id") for r in runs] or [], key="rd_run_sel")
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
            except APIError:
                cruns = []
        if cruns:
            st.dataframe(pd.DataFrame([{"id": r.get("id")[:8], "status": r.get("status"), "created_at": r.get("created_at")} for r in cruns]), use_container_width=True, hide_index=True)
            sel = st.selectbox("Select code run", options=[r.get("id") for r in cruns] or [], key="code_run_sel")
            if sel and st.button("Show code artifact"):
                try:
                    det = get_code_run(api_url, api_token, sel)
                    st.json(det)
                except APIError as exc:
                    st.error(str(exc))
        else:
            st.caption("No code runs yet.")
