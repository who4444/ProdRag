# ProdRag frontend (Streamlit)

Standard UI: upload PDFs, track async ingestion, and ask questions with a
streamed answer plus retrieved text/table sources and figure thumbnails.

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

- Ingestion is async: uploads are queued and status is shown in the Documents
  tab (refresh to poll).
- `/query` streams NDJSON; the answer is rendered progressively. Figure sources
  are fetched via the authenticated `/files` endpoint and shown inline.
