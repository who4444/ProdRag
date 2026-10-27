# ProdRag frontend (Streamlit)

`app.py` is a single-page full-run UI: describe one idea, press **Run**, and the
whole pipeline executes in place. Prior tab features live under a collapsed
**🧰 Utilities** expander.

## Flow

- One idea → inline clarification to lock requirements.
- Research streams live into a `ResearchArtifact`.
- Code stage picks a mode: **Demo PoC** (`/code/generate`), **Extend a codebase**
  (`/code/extend`, repo index/select + optional feasibility pre-check), or
  **Both** — auto-chained behind the single Run button.
- Consolidated **Result** section with per-file download buttons.
- Documents (upload/status/delete/reingest), Ask, Memory, and Runs history stay
  available in the Utilities expander.

## Run

```bash
uv sync --extra frontend
cd frontend
uv run streamlit run app.py          # http://localhost:8501
```

Or with plain pip:

```bash
pip install -r frontend/requirements.txt
cd frontend && streamlit run app.py
```

## Configuration

Set these via the sidebar, `STREAMLIT` secrets, or env vars:

| Setting                  | Env var                | Default              |
| ------------------------ | ---------------------- | -------------------- |
| Backend URL              | `PRODRAG_API_URL`      | `http://localhost:8000` |
| Bearer token             | `PRODRAG_API_TOKEN`    | `change-me`          |

The sidebar "Check connection" hits `/health` to validate before you start.

## Notes

- Ingestion is async: uploads are queued and status is shown under Utilities →
  Documents (refresh to poll).
- `/query` streams NDJSON; the answer is rendered progressively. Figure sources
  are fetched via the authenticated `/files` endpoint and shown inline.
- `simple_test_app.py` and `simple_test.html` remain as minimal API test
  harnesses (not the main UI).
