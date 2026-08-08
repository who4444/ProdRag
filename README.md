# ProdRag

Production-grade multimodal RAG. Ingests PDFs (text + figures/tables), indexes text and
vision embeddings in separate vector spaces, and answers with a vision LLM grounded on both.

## Architecture

```
                ┌──────────────┐   enqueue job    ┌──────────────┐
   PDF upload ─►│  api (FastAPI)├────────────────►│ worker (arq) │
                └──────┬───────┘                  └──┬───────────┘
                       │                             │ parse (PyMuPDF)
                       ▼                             ▼
                   MinIO/S3                     Qdrant (text + image)
                 raw PDF, figures              Redis (queue, status)
                       │
                       ▼
                   ┌────────────┐
      query ──────►│  api /query │──► text-embedding-3 + CLIP query encode
                   └─────┬──────┘──► retrieve text + images ──► gpt-4o (vision) ─► NDJSON stream
```

- **Ingestion**: `POST /documents` stores the PDF in MinIO, enqueues a job. The worker
  parses pages (PyMuPDF), chunks text (langchain splitter), extracts embedded figures,
  stores figures in MinIO, embeds text with **bge-m3 on a Modal GPU**
  (`deploy/modal/embed_service.py`) and figures with **CLIP on a Modal GPU**
  (`deploy/modal/clip_service.py`), then indexes into two Qdrant collections.
- **Retrieval**: text chunks searched with the OpenAI embedding of the question; figures
  searched with the CLIP text-embedding of the question (shared CLIP vision-text space).
- **Answering**: retrieved text + figure images are passed to the chat model
  (`deepseek-v4-flash` by default, OpenAI-compatible) and streamed back as NDJSON
  (`sources` event, then `text` deltas). DeepSeek is chat-only, so figures are
  returned as `image_url` references in sources but not attached to the model;
  set `PRODRAG_CHAT_SUPPORTS_IMAGES=true` with a vision model (e.g. gpt-4o) to
  ground the answer on the figures directly.

## Run

```bash
cp .env.example .env   # set PRODRAG_CHAT_API_KEY (DeepSeek) and PRODRAG_API_TOKEN
docker compose -f deploy/docker-compose.yml up --build
curl localhost:8000/health
```

## Local dev / tests

```bash
uv sync --extra dev
uv run pytest
uv run uvicorn app.main:app --reload   # from backend/, needs local qdrant/redis/minio
```

## API

All endpoints require `Authorization: Bearer $PRODRAG_API_TOKEN`.

- `POST /documents` — multipart `file` + optional `metadata` (JSON string). Returns
  `{"document_id", "job_id"}`; ingestion runs async.
- `GET /documents/{id}/status` — `{"status": "processing|done|error", "text_chunks", "images", "error"}`
- `POST /query` — `{"question": "...", "k": 5, "k_images": 5}` → NDJSON stream:
  `{"type":"sources","items":[...]}` then `{"type":"text","delta":"..."}`. Image sources
  include an `image_url` you can fetch via `GET /files/{object_key}`.
- `GET /files/{object_key}` — fetch a stored figure (proxies MinIO).

## GPU services on Modal

Text (bge-m3) and vision (CLIP) embeddings run as GPU services on Modal so the
API/worker stay torch-free.

```bash
uv sync --extra dev --extra modal
modal secret create prodrag-clip-token AUTH_TOKEN="$(openssl rand -hex 24)"   # once
modal secret create prodrag-embed-token AUTH_TOKEN="$(openssl rand -hex 24)"   # once
modal deploy deploy/modal/embed_service.py
modal deploy deploy/modal/clip_service.py
# set PRODRAG_EMBED_SERVICE_URL/_TOKEN and PRODRAG_CLIP_SERVICE_URL/_TOKEN in .env
```

See `deploy/modal/README.md`. Without the URLs set, the backend falls back to
OpenAI `text-embedding-3-small` (set `PRODRAG_EMBEDDING_DIM=1536`) and a local
torch load for CLIP.

## Eval

```bash
PRODRAG_API_TOKEN=... python eval/recall.py
```

## Notes

- Embedding services run on Modal GPU. Without `PRODRAG_EMBED_SERVICE_URL` text
  embeddings fall back to OpenAI (`PRODRAG_EMBEDDING_DIM=1536`); without
  `PRODRAG_CLIP_SERVICE_URL` image retrieval is skipped.
- Changing the text embedding model changes `PRODRAG_EMBEDDING_DIM`; delete and
  re-ingest (or drop the qdrant volume) so the collection is recreated at the
  right dimension.
- Re-ingesting the same PDF overwrites points (deterministic Qdrant ids) — idempotent.
