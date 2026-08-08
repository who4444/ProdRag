# Modal GPU services — embeddings for ProdRag

ProdRag's GPU workloads (text + vision embeddings) run here as Modal Apps with
T4 GPUs and are called by the CPU-side API/worker over HTTPS — so the backend
images never bundle torch.

- `clip_service.py` — CLIP `ViT-B-32` vision embeddings (512 dims).
- `embed_service.py` — `BAAI/bge-m3` text embeddings (1024 dims).

## Deploy

Prereqs: `modal` CLI installed and authenticated (`modal setup`), Python ≥3.12.

```bash
uv sync --extra dev --extra modal
modal secret create prodrag-clip-token AUTH_TOKEN="$(openssl rand -hex 24)"   # once
modal secret create prodrag-embed-token AUTH_TOKEN="$(openssl rand -hex 24)"   # once
modal deploy deploy/modal/clip_service.py
modal deploy deploy/modal/embed_service.py
```

Each deploy prints its endpoint URL.

## Wire the backend

```bash
# .env
PRODRAG_EMBED_SERVICE_URL=https://<workspace>--prodrag-embed-textembedder-embed.modal.run
PRODRAG_EMBED_SERVICE_TOKEN=<embed AUTH_TOKEN>
PRODRAG_CLIP_SERVICE_URL=https://<workspace>--prodrag-clip-clipembedder-embed.modal.run
PRODRAG_CLIP_SERVICE_TOKEN=<clip AUTH_TOKEN>
```

`backend/app/embeddings/text.py` prefers the remote bge-m3 service and falls
back to OpenAI `text-embedding-3-small` when `PRODRAG_EMBED_SERVICE_URL` is
unset (then set `PRODRAG_EMBEDDING_DIM=1536`). `vision.py` prefers the remote
CLIP service and only falls back to a local torch load if the URL is unset.

## Requests

`POST {URL}/` (the deploy URL is the full endpoint) with a Bearer token.

bge-m3 (`embed_service.py`), queries get the retrieval instruction prefix:

```json
{"data": ["what does the latency chart show?"], "query": true}
{"data": ["passage text ..."], "query": false}
```

CLIP (`clip_service.py`):

```json
{"type": "image", "data": ["<base64 png>", ...]}
{"type": "text", "data": ["what does the latency chart show?"]}
```

Responses are `{"embeddings": [[...], [...]]}`, L2-normalized, and (for text)
include `model` and `dim` so the client can verify `PRODRAG_EMBEDDING_DIM`.

## Changing the embedding model

The Qdrant collection is created with `PRODRAG_EMBEDDING_DIM` on first startup.
If you switch models, delete/recreate the `text_chunks` collection (e.g. drop
the qdrant volume) or reindex.

## Cost / scaling

- `min_containers=1`, `max_containers=4`, `concurrent(max_inputs=8)`: one warm T4,
  up to 4 total, each handling 8 concurrent batch requests. Batch, don't stream
  single images, to keep T4 utilization high.
- Swap `gpu="T4"` for `gpu="A10G"`/`"L4"` if throughput needs grow.
- GPU memory is idle when the worker has no pending figures — Modal scales the
  pool to zero (minus the warm container).
