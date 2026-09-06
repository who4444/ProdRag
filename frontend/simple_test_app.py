"""Minimal Streamlit test frontend — alternative to simple_test.html.

Run: cd frontend && uv run streamlit run simple_test_app.py
Covers: health, documents, RAG, R&D analyzer+research, web fallback.
"""

import json

import streamlit as st

from api import APIError, document_status, get_file, health, query_stream, rd_analyze, rd_research_stream, upload_document

st.set_page_config(page_title="ProdRag Simple Test", page_icon="🧪", layout="wide")
st.title("🧪 ProdRag — Simple Test")

api_url = st.text_input("API URL", value="http://localhost:8000")
api_token = st.text_input("API token", type="password", value="change-me")

if st.button("Check health"):
    try:
        st.json(health(api_url))
        st.success("OK")
    except Exception as e:
        st.error(str(e))

tab_docs, tab_rag, tab_rd = st.tabs(["Documents", "RAG", "R&D"])

with tab_docs:
    f = st.file_uploader("PDF", type=["pdf"])
    title = st.text_input("Title")
    if st.button("Upload", disabled=f is None):
        try:
            meta = {"title": title} if title else {}
            res = upload_document(api_url, api_token, f.name, f.getvalue(), meta)
            st.json(res)
        except APIError as e:
            st.error(str(e))
    if st.button("List docs"):
        try:
            import requests
            r = requests.get(f"{api_url}/documents", headers={"Authorization": f"Bearer {api_token}"}, timeout=10)
            r.raise_for_status()
            st.json(r.json())
        except Exception as e:
            st.error(str(e))

with tab_rag:
    q = st.text_area("Question", placeholder="What is the main contribution?")
    k = st.slider("k", 1, 10, 4)
    ki = st.slider("k_images", 0, 10, 2)
    if st.button("Ask", disabled=not q.strip()):
        try:
            resp = query_stream(api_url, api_token, q, k, ki)
            with resp:
                lines = resp.iter_lines(decode_unicode=True)
                first = json.loads(next(lines))
                st.write(f"Sources: {len(first.get('items', []))}")
                for it in first.get("items", []):
                    if it["kind"] == "image":
                        try:
                            img = get_file(api_url, api_token, it["image_url"].removeprefix("/files/"))
                            st.image(img, caption=f"page {it['page']} score {it['score']}", width=250)
                        except APIError:
                            pass
                    else:
                        st.markdown(f"[page {it['page']}] {it['content']} — *{it['score']}*")
                st.divider()
                def deltas():
                    for line in lines:
                        if line:
                            evt = json.loads(line)
                            if evt.get("type") == "text":
                                yield evt.get("delta", "")
                st.write_stream(deltas())
        except APIError as e:
            st.error(str(e))

with tab_rd:
    idea = st.text_area("Idea / paper", placeholder="Turn Attention Is All You Need into a minimal demo")
    if "simple_req" not in st.session_state:
        st.session_state.simple_req = None
    if st.button("Analyze", disabled=not idea.strip()):
        try:
            res = rd_analyze(api_url, api_token, idea)
            st.json(res)
            if res.get("ready"):
                st.session_state.simple_req = res["requirements"]
                st.success("Requirements ready")
            else:
                st.session_state.simple_req = None
                st.warning(f"Need clarification: {res.get('questions')}")
                # simple inline answers
                answers = []
                for i, qq in enumerate(res.get("questions", [])):
                    ans = st.text_input(f"Q{i+1}: {qq}", key=f"q{i}")
                    answers.append(ans)
                if st.button("Submit answers"):
                    res2 = rd_analyze(api_url, api_token, idea, answers=[a for a in answers if a.strip()])
                    st.json(res2)
                    if res2.get("ready"):
                        st.session_state.simple_req = res2["requirements"]
        except APIError as e:
            st.error(str(e))

    req = st.session_state.simple_req
    if req:
        st.json(req)
        if st.button("Start R&D research"):
            try:
                resp = rd_research_stream(api_url, api_token, req, idea=idea)
                with resp:
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        evt = json.loads(line)
                        if evt["type"] == "orchestrator":
                            st.write(f"Plan: {len(evt['directions'])} directions")
                        elif evt["type"] == "subagent":
                            if evt["event"] == "start":
                                st.caption(f"🔎 {evt['direction_id']}")
                            else:
                                st.write(f"**{evt['summary']['direction_id']}** ({evt['summary']['confidence']}) {evt['summary']['findings'][:300]}")
                        elif evt["type"] == "validator":
                            st.json(evt["validation"])
                        elif evt["type"] == "artifact":
                            st.success("Artifact ready")
                            st.json(evt["data"]["demo_spec"])
                            with st.expander("Full artifact"):
                                st.json(evt["data"])
            except APIError as e:
                st.error(str(e))
