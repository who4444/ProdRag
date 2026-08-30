# ProdRag — R&D Demo System

Turns research papers and ideas into **runnable demo artifacts** — a `demo.py` + `requirements.txt` that demonstrates a paper's core idea on toy data.

> **Pipeline:** `idea -> research -> code`. The existing multimodal RAG stack is packaged as a reusable tool for both the chat UI and the research pipeline. See **[PROJECT.md](PROJECT.md)** for the living spec, data model, and roadmap.

## Architecture

```
              ┌─────────────────────────────────────────────────┐
              │              FastAPI (api)                       │
              │  /rd/analyze  /rd/research  /query  /research    │
              │  /documents  /files  /health                     │
              └──────────────┬──────────────────────────────────┘
                             │
   ┌─────────────────────────┼─────────────────────────┐
   │  R&D Pipeline (new)     │        RAG Tool          │  Legacy Research
   │  Analyzer ──►           │  search_kb()            │  agent (single-loop
   │  Orchestrator ──►       │  hybrid dense+BM25      │  kb_search loop)
   │  Subagents x5 ──►       │  + rerank + CLIP        │
   │  Validator ──► Artifact │  (tools/rag.py)         │
   └──────────┬──────────────┴──────────┬──────────────┘
              │                         │
              ▼                         ▼
     Supabase Postgres          Qdrant (text_chunks, image_chunks, episodes)
     (documents, parts,        Redis (queue, conv:{session})
      research_runs,           Supabase Storage (PDFs, figures)
      artifacts, conv)         Modal GPU (bge-m3, CLIP) + DeepSeek chat
```

* **RAG Tool** (`backend/app/tools/rag.py`): thin wrapper over `backend/app/rag/retrieval.py:search_kb` — hybrid dense (bge-m3) + sparse BM25 via RRF, optional cross-encoder rerank, plus CLIP image retrieval. Shared by `/query`, the legacy `/research` agent, and the new R&D subagents. No duplication.
* **Analyzer** (`backend/app/rd/analyzer.py`): stateless, second-request handshake. Takes `idea + answers`, returns `Requirements{goal, core_idea, demo_type, constraints, ...}` or `clarify` questions. **No KB access** — keeps it cheap. Max 2 rounds.
* **Orchestrator** (`backend/app/rd/orchestrator.py`): decomposes `Requirements` into a fixed 5-direction plan — `core_algorithm`, `datasets`, `baselines`, `implementation_details`, `evaluation` — and fans out in parallel.
* **Subagents** (`backend/app/rd/subagent.py`): one function, N concurrent invocations. Each does `rag_search` + LLM synthesis into a `ResearchSummary{findings, sources, confidence, gaps}`.
* **Validator** (`backend/app/rd/validator.py`): LLM-as-judge comparing `Requirements` vs. collected summaries. Always emits an artifact with `risks` when feasible — never blocks. At most one re-search iteration.
* **Artifact** (`backend/app/rd/artifacts.py`): handoff to the coding module — `demo_spec{entrypoint, tech_stack, core_algorithm, pseudocode, acceptance_criteria}` + deduped sources.

v1 demo artifact is intentionally minimal: `demo.py` + `requirements.txt` on toy data. Gradio/Docker are v2.

## Flow

```
User idea (free text)  ──POST /rd/analyze──►  {ready:false, questions:[...]}
                                ▲                    │
                          answers│                    │ ready:true, requirements
                                └───POST /rd/analyze─┘
                                              │
                                   POST /rd/research {requirements, session_id}
                                              │  NDJSON stream:
                                              │  analyzer/requirements
                                              │  orchestrator/plan
                                              │  subagent/start, subagent/result (x5)
                                              │  validator/result
                                              │  artifact
                                              ▼
                                       Research Artifact
                                              │
                                              ▼
                                       Coding Module (next)
```

If the idea is already precise, the first `/rd/analyze` returns `ready:true` immediately — no second request.

## Run

```bash
cp .env.example .env   # set PRODRAG_SUPABASE_URL/_SECRET_KEY, PRODRAG_CHAT_API_KEY, PRODRAG_API_TOKEN
docker compose -f deploy/docker-compose.yml up --build
curl localhost:8000/health
```

Storage is **Supabase Storage** via native REST. From Supabase dashboard: Project Settings > API gives the project URL and secret key. Create the `prodrag-assets` bucket there (dashboard or `insert into storage.buckets (id, name, public) values ('prodrag-assets', 'prodrag-assets', false);`). Apply `backend/app/core/schema.sql` via SQL editor or `psql` before first run (includes `research_runs` / `research_artifacts`).

## Local dev / tests

```bash
uv sync --extra dev
uv run pytest
uv run uvicorn app.main:app --reload   # from backend/, needs local qdrant/redis + Supabase
```

## Frontend (Streamlit)

```bash
uv sync --extra frontend
cd frontend && uv run streamlit run app.py   # http://localhost:8501
```

Four tabs: **Documents** (upload PDFs, track async ingestion), **Ask** (streamed RAG answer), **Research** (legacy single-loop agent), **R&D** (new pipeline — analyzer clarifying questions, live subagent progress, validator risks, downloadable artifact spec).

Sidebar: `PRODRAG_API_URL`, `PRODRAG_API_TOKEN`.

## API

All endpoints require `Authorization: Bearer $PRODRAG_API_TOKEN`.

| Endpoint | Description |
|---|---|
| `GET /health` | liveness |
| `POST /documents` | multipart `file` + `metadata` JSON. Returns `{document_id, job_id}` or `{document_id, duplicate:true}` on SHA-256 dedup |
| `POST /documents/{id}/reingest` | re-enqueue ingest (retry after error, cleans old points/figures first) |
| `GET /documents/{id}/status` | `{"status","text_chunks","images","error"}` |
| `GET /documents` / `DELETE /documents/{id}` | list / delete |
| `POST /query` | `{question, k, k_images}` → NDJSON `sources` + `text` deltas |
| `POST /research` | legacy agent: `{question, session_id?, k, k_images}` → NDJSON `agent/tool_call`, `sources`, `text`, `memory/*` |
| `POST /rd/analyze` | `{idea, session_id?, answers?: string[]}` → `{ready, requirements?, questions?}`. Second-request handshake; analyzer has no KB access |
| `POST /rd/research` | `{requirements, session_id?, idea?}` → NDJSON `orchestrator/plan`, `subagent/*` (x5, parallel), `validator/result`, `artifact`. Fixed taxonomy: `core_algorithm, datasets, baselines, implementation_details, evaluation`. Validator emits with risks, never blocks |
| `GET /files/{object_key}` | fetch a stored figure via Supabase Storage |

NDJSON streaming everywhere interactive (`application/x-ndjson`).

## GPU services on Modal

Text (bge-m3) and vision (CLIP) run on Modal so api/worker stay torch-free.

```bash
uv sync --extra dev --extra modal
modal secret create prodrag-clip-token AUTH_TOKEN="$(openssl rand -hex 24)"   # once
modal secret create prodrag-embed-token AUTH_TOKEN="$(openssl rand -hex 24)"   # once
modal deploy deploy/modal/embed_service.py
modal deploy deploy/modal/clip_service.py
# set PRODRAG_EMBED_SERVICE_URL/_TOKEN and PRODRAG_CLIP_SERVICE_URL/_TOKEN in .env
```

See `deploy/modal/README.md`. Without URLs set, fallback to OpenAI `text-embedding-3-small` (`PRODRAG_EMBEDDING_DIM=1536`) and local CLIP. Services scale to zero (`min_containers=0`). Manage: `bash deploy/modal/manage.sh {status|stop|start}`.

## Eval

```bash
PRODRAG_API_TOKEN=... python eval/recall.py
```

Planned: faithfulness / citation-coverage eval for R&D artifacts (beyond recall@k).

## Notes

* Modal GPU URLs absent → fallback embeddings; `PRODRAG_EMBEDDING_DIM` must match the collection.
* `uv sync --extra dev --extra modal --extra frontend` (all three; single-extra drops the others).
* `ensure_collections()` swallows 409 — safe for concurrent api + worker boot.
* Deterministic Qdrant point IDs → idempotent re-ingestion; re-ingest cleans orphans.
