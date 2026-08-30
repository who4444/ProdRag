# ProdRag — R&D Demo System (Project Spec)

Living spec for the ProdRag project. Read this first, then `README.md` for run instructions. This document describes the rescope from a *research assistant* into an **R&D demo system** that turns research papers/ideas into **runnable artifacts** (`demo.py` + `requirements.txt`) via a staged research pipeline.

---

## 1. Vision

A production-grade R&D demo factory. Given a paper or research idea, it:

1. **Analyzes** the idea into structured `Requirements` — goal, core idea, demo type, constraints — asking 1–3 clarifying questions when the input is underspecified.
2. **Researches** — an orchestrator decomposes requirements into 5 fixed directions and fans out to subagents that each gather evidence from the knowledge base and synthesize a summary.
3. **Validates** — compares requirements vs. collected evidence, flagging gaps/risks (always emits an artifact, never blocks).
4. **Materializes** — emits a `ResearchArtifact` (`demo_spec + sources + acceptance_criteria`) that the coding module turns into a runnable demo on toy data.

```
idea -> [Analyzer] -> Requirements -> [Orchestrator -> Subagents x5] -> [Validator] -> Artifact -> [Coding module]
```

### Non-goals (for now)

- No web search integration (phase 2, optional).
- No fine-tuning; all model behavior is prompt/tool-driven.
- No user accounts/auth beyond the single `PRODRAG_API_TOKEN`.
- DeepSeek is the chat model and is **text-only** (no vision, no embeddings API).
- v1 demo is a minimal `demo.py` script (or notebook) on toy data — Gradio/Docker are v2.

---

## 2. Current state (verified 2026-08-30)

The system is a working multimodal RAG pipeline extended with the R&D research pipeline. Everything below is running/verified except where noted.

### 2.1 Stack

| Concern        | Technology | Notes |
|---------------|-----------|-------|
| API            | FastAPI (`backend/app/main.py`) | NDJSON streaming, Bearer token auth |
| Async ingestion| arq + Redis | worker `backend/app/worker.py`, job in `backend/app/ingest/pipeline.py` |
| Vector store   | Qdrant (`deploy/docker-compose.yml`, pinned `v1.19.0`) | 3 collections, see §2.3 |
| Object storage | Supabase Storage native REST API (`backend/app/core/storage.py`) | secret API key, httpx; no boto3/S3 |
| Relational DB  | Supabase Postgres (`backend/app/core/db.py`) | documents, parts, conversations, messages, episodes, research_runs, research_artifacts (see §2.3b) |
| Chat LLM       | DeepSeek `deepseek-v4-flash` (OpenAI-compatible, `chat_base_url=https://api.deepseek.com`) | text-only → `chat_supports_images=false` |
| Text embeddings| **Modal GPU** bge-m3 `BAAI/bge-m3` (1024 dims), `deploy/modal/embed_service.py` | remote-first via httpx in `backend/app/core/embeddings/text.py`; OpenAI fallback |
| Vision embeddings | **Modal GPU** CLIP `ViT-B-32` (512 dims), `deploy/modal/clip_service.py` | remote-first in `backend/app/core/embeddings/vision.py`; local torch fallback |
| PDF parsing    | PyMuPDF (`backend/app/ingest/parser/pdf.py`) | text + page render + embedded figures |
| RAG Tool       | thin wrapper `backend/app/tools/rag.py` over `backend/app/rag/retrieval.py` | shared by /query, legacy /research, and R&D subagents |
| R&D Pipeline   | `backend/app/rd/{analyzer,orchestrator,subagent,validator,artifacts,pipeline}.py` | idea -> research -> artifact |
| Frontend       | Streamlit (`frontend/app.py`) | Documents + Ask + Research + R&D tabs |
| Eval           | `eval/recall.py` (Recall@k vs golden set) | artifact faithfulness planned |

### 2.2 Repo layout

```
backend/app/           # FastAPI + worker package (organized by concern)
  main.py              # app wiring: lifespan, /health, mounts api router
  config.py            # pydantic-settings, env prefix PRODRAG_
  schemas.py           # QueryRequest, ResearchRequest + R&D schemas
  api/                 # HTTP layer: routers per resource
    __init__.py        # router assembly + require_token auth
    documents.py       # upload / status / reingest
    files.py           # /files/{object_key} proxy
    query.py           # /query
    research.py        # /research (legacy single-loop agent)
    rd.py              # /rd/analyze, /rd/research (new pipeline)
  core/                # infrastructure
    storage.py         # Supabase Storage REST client (httpx, secret API key)
    db.py              # Supabase Postgres CRUD (documents, parts, conv, episodes, research_runs)
    schema.sql         # DDL for the relational layer (apply via SQL editor / psql)
    vectorstore.py     # AsyncQdrantClient, ensure_collections, search/upsert (+episodes)
    embeddings/{text,vision}.py
    ocr.py             # Tesseract via pytesseract (graceful degradation)
    retry.py           # retry_async (exp backoff, transient-predicate)
  tools/
    rag.py             # rag_search() wrapper — the packaged RAG tool
  rd/                  # R&D pipeline (idea -> research -> artifact)
    schemas.py         # Requirements, Direction, ResearchSummary, Validation, ResearchArtifact
    analyzer.py        # second-request handshake, no KB
    orchestrator.py    # fixed 5-direction fan-out plan
    subagent.py        # kb_search + synthesis per direction
    validator.py       # LLM-as-judge, emit-with-risks
    artifacts.py       # demo_spec + acceptance_criteria builder
    pipeline.py        # rd_research() — top-level orchestrator
  ingest/              # ingestion pipeline
    pipeline.py        # arq job ingest_document (parse -> embed -> index)
    chunking.py        # langchain text splitter
    parser/pdf.py      # PyMuPDF: text + figures + tables
  rag/                 # RAG layer (retrieval + answering)
    retrieval.py       # search_kb() + source_items() — shared by /query and the agent tool
    answer.py          # answer(): sources + text deltas (NDJSON)
  agent.py             # research(): single tool-calling loop (legacy, kept)
  memory.py            # conv (Supabase messages) + episodic (Qdrant + Supabase)
  worker.py            # arq WorkerSettings
backend/tests/         # test_chunking.py, test_pdf_parser.py, test_agent.py, test_rd_*.py
deploy/docker-compose.yml   # qdrant + redis + api + worker (no MinIO anymore)
deploy/modal/          # embed_service.py, clip_service.py, manage.sh, README.md
frontend/              # Streamlit app.py + api.py (Documents / Ask / Research / R&D tabs)
eval/                  # recall.py + golden.jsonl
PROJECT.md             # this file
```

### 2.3 Qdrant collections

| Collection    | Dim | Vector | Payload (per point) |
|-------------|-----|--------|---------------------|
| `text_chunks` | 1024| bge-m3 dense + **BM25 sparse (`bm25`)** | `doc_id, source, page, chunk_idx, kind, metadata, text` |
| `image_chunks`| 512 | CLIP   | `doc_id, source, page, image, object_key, caption, ocr_text, metadata` |
| `episodes`   | 1024| bge-m3 | `session_id, created_at, question, answer, sources[]` |

- Point IDs are **deterministic** (SHA-256 of kind + payload) → re-ingesting the same PDF is idempotent (upsert overwrites).
- Changing the text embedding model changes `PRODRAG_EMBEDDING_DIM` → collection must be recreated (drop qdrant volume or delete collection) and data re-ingested.
- `text_chunks` is hybrid (dense + BM25 sparse, RRF-fused in `search_text`); a pre-existing collection without the `bm25` sparse vector warns at boot and searches fall back to dense until the collection is deleted and re-ingested.

### 2.3b Relational layer (Supabase Postgres)

System of record for entities & lifecycle; Qdrant stays vector-only. DDL lives in `backend/app/core/schema.sql` (apply via SQL editor or `psql` before running).

| Table | Purpose |
|-------|---------|
| `documents` | lifecycle/status, title, source_key, content_hash, metadata, chunk counts |
| `parts`     | one row per Qdrant point (`id` = point id) → doc → parts join enables delete-doc |
| `conversations` / `messages` | durable chat history (replaces Redis `conv:` as source of truth) |
| `episodes`  | episodic memory metadata (vector stays in Qdrant `episodes`) |
| `research_runs` | R&D pipeline runs: `id, session_id, requirements jsonb, status, created_at` |
| `research_artifacts` | materialized artifacts: `run_id fk, artifact jsonb, validation jsonb, created_at` |

Env: `PRODRAG_DATABASE_URL` (Postgres connection string). Driver: `supabase` REST client (`core/db.py`). Endpoints: `GET /documents`, `DELETE /documents/{id}`, `/rd/*`.

### 2.4 API (current)

All endpoints require `Authorization: Bearer $PRODRAG_API_TOKEN`.

- `GET  /health`
- `POST /documents` — multipart `file` + optional `metadata` (JSON str) → `{document_id, job_id}` (async ingest). Dedups by SHA-256 of the file: an identical previously-ingested upload returns `{document_id, job_id: null, duplicate: true}` instead of re-ingesting.
- `POST /documents/{id}/reingest` — re-enqueues the ingest job for a document (retry after `error`, or refresh a changed source). Re-ingestion deletes the old Qdrant points + stored figures first, so nothing is orphaned.
- `GET  /documents/{id}/status` — `{status, text_chunks, images, error}`
- `GET  /documents` / `DELETE /documents/{id}` — list / delete
- `POST /query` — `{question, k, k_images}` → NDJSON stream: `{"type":"sources","items":[...]}` then `{"type":"text","delta":"..."}`
- `POST /research` — legacy single-loop agent: `{question, session_id?, k, k_images}` → NDJSON stream: `memory/session`, `agent/tool_call`, `sources`, `agent/tool_result`, `text` (final answer), `memory/saved`.
- `POST /rd/analyze` — stateless second-request handshake: `{idea, session_id?, answers?: string[]}` → `{ready: bool, requirements?: Requirements, questions?: string[]}`. Analyzer has **no KB access** — max 2 rounds. First call without `answers` may return `ready:false` with 1–3 clarifying questions; client re-POSTs with `answers`.
- `POST /rd/research` — R&D pipeline: `{requirements, session_id?, idea?}` → NDJSON stream: `orchestrator/plan`, `subagent/start`, `subagent/result` (x5, parallel), `validator/result`, `artifact`. Fixed taxonomy: `core_algorithm`, `datasets`, `baselines`, `implementation_details`, `evaluation`. Validator always emits an artifact (emit-with-risks); never blocks.
- `GET  /files/{object_key:path}` — fetch a stored figure (proxies storage)

### 2.5 Env inventory (`PRODRAG_` prefix)

`api_token`, `chat_api_key`, `chat_base_url`, `chat_model`,
`chat_supports_images`, `max_images_in_context`, `openai_api_key`,
`embedding_model`, `embedding_dim`, `embed_service_url`,
`embed_service_token`, `vision_embedding_model`, `vision_embedding_pretrained`,
`vision_dim`, `clip_service_url`, `clip_service_token`, `qdrant_url`,
`qdrant_api_key`, `collection_text`, `collection_image`, `redis_url`,
`supabase_url`, `supabase_secret_key`,
`storage_bucket`, `database_url`, `chunk_size`, `chunk_overlap`, `page_dpi`, `min_figure_area`,
`retrieval_hybrid_k`, `rerank_service_url`, `rerank_service_token`, `rerank_model`.

Two `.env` files: repo-root `.env` (used by docker compose) and `backend/.env` (used when running uvicorn from `backend/`). Both gitignored.

### 2.6 Modal GPU services

- Deployed apps: `prodrag-embed`, `prodrag-clip`, `prodrag-rerank` (T4 GPUs).
- **Scale to zero** (`min_containers=0`): idle/offline server costs $0.
- `deploy/modal/manage.sh {status|stop|start}` for explicit control.
- Cold start ~30-60s; client timeouts in `embeddings/*.py` are 120s.
- Bearer tokens stored in Modal secrets `prodrag-embed-token`, `prodrag-clip-token`; values mirrored in `.env`.

### 2.7 Operational gotchas (learned)

- `uv sync` with specific `--extra`s **drops the others**. Always sync all: `uv sync --extra dev --extra modal --extra frontend`.
- Qdrant client `1.19.x` renamed `search()` → `query_points()`.
- `ensure_collections()` races between api and worker at boot; it must swallow the 409 `UnexpectedResponse` (already handled in `vectorstore.py`).
- Docker-compose must set internal `PRODRAG_QDRANT_URL`/`PRODRAG_REDIS_URL` via `environment` (localhost defaults fail inside containers).
- **Supabase storage creds are not yet in `.env`** — the stack currently can't boot until `PRODRAG_SUPABASE_URL` and `PRODRAG_SUPABASE_SECRET_KEY` are set and the `prodrag-assets` bucket exists.
- Streamlit runs from `frontend/`; `api.py` is imported relative to the app dir.

---

## 3. Target architecture — R&D demo system

```
                         ┌──────────────────────────────────────────────────┐
                         │                 FastAPI (api)                     │
                         │  /rd/analyze  /rd/research  /query  /research    │
                         │  /documents  /files  /health                     │
                         └───────┬──────────────────────┬───────────────────┘
                                 │                      │
               ┌─────────────────┘                      └─────────────────┐
               │ R&D Pipeline (idea -> research -> artifact)              │ Legacy / general
               │                                                          │
   ┌───────────▼───────────┐    ┌──────────────────┐    ┌───────────────▼────┐
   │ Analyzer              │    │ Orchestrator      │    │ agent.py           │
   │ no KB, 2-req handshake│───►│ fixed 5-direction │───►│ single-loop        │
   │ Requirements or       │    │ plan              │    │ kb_search loop     │
   │ clarify questions     │    └────────┬─────────┘    └────────────────────┘
   └───────────────────────┘             │ fan-out (asyncio.gather)
                              ┌──────────┼──────────┐
                              │          │          │
                       ┌──────▼──┐ ┌────▼────┐ ┌───▼─────┐  x5
                       │Subagent │ │Subagent │ │Subagent │  each: kb_search
                       │core_algo│ │datasets │ │  ...    │  + synthesis
                       └────┬────┘ └────┬────┘ └───┬─────┘
                            └───────────┼──────────┘
                                        ▼
                              ┌──────────────────┐
                              │ Validator        │  LLM-as-judge
                              │ emit-with-risks  │  coverage + feasibility
                              └────────┬─────────┘
                                       ▼
                              ┌──────────────────┐
                              │ ResearchArtifact │  demo_spec + sources +
                              │ (handoff to     │  acceptance_criteria
                              │  coding module)  │
                              └──────────────────┘

     RAG Tool (tools/rag.py) ──► search_kb() [hybrid dense+BM25 + rerank + CLIP]
     Memory (memory.py) ───────► conv:{session} + episodes (Qdrant)
     Storage ──────────────────► Qdrant (text_chunks, image_chunks, episodes)
                                 Redis (queue, conv) / Supabase Storage / DB
```

**Layers:**

1. **RAG Tool** (packaged) — `backend/app/tools/rag.py` wraps `backend/app/rag/retrieval.py:search_kb`. Used by `/query`, legacy `/research`, and all R&D subagents. No duplication.
2. **R&D Pipeline** — `backend/app/rd/` — staged: analyzer -> orchestrator -> subagents -> validator -> artifact.
3. **Memory** — conversation (per session, `conv:{id}`) + episodic (`episodes` collection).
4. **Coding module** (next) — consumes `ResearchArtifact` and emits `demo.py + requirements.txt` (v1: script on toy data).

---

## 4. R&D pipeline design

### 4.1 Analyzer — `backend/app/rd/analyzer.py`

Stateless, no KB access, second-request handshake. Reuses `memory.py` for optional session continuity but does not retrieve.

* **Input:** `AnalyzeRequest{idea: str, session_id?: str, answers?: list[str]}`. `idea` is the raw user input; `answers` are replies to prior clarifying questions.
* **Output:** `{ready: bool, requirements?: Requirements, questions?: list[str], draft?: Requirements}`.
* **Requirements shape:**
```python
class Requirements(BaseModel):
    goal: str               # what the demo must do
    core_idea: str          # paper's core idea in 1-2 sentences
    demo_type: str          # script | notebook  (v1 enum; gradio/api are v2)
    constraints: list[str]  # e.g. ["python-only", "no GPU"]
    audience: str           # researcher | student | engineer
    open_questions: list[str]
```
* **Logic:** single LLM call (`settings.chat_model`, `response_format=json_object`). If `idea + answers` is underspecified, return `ready=false` with 1–3 questions (e.g. "what dataset?", "what interaction?"). Cap at 2 rounds — second call with answers must return `ready=true` (infer defaults afterward).
* **Streaming:** `/rd/analyze` is **not** streamed — it's a plain JSON response (fast, <2s). The pipeline's NDJSON stream starts at `/rd/research`.
* **Ponytail:** no LangGraph, no tool calls, no KB — just one prompt + structured output.

### 4.2 Orchestrator — `backend/app/rd/orchestrator.py`

Decomposes `Requirements` into a fixed 5-direction plan.

* **Fixed taxonomy (no LLM freedom):**

| id | question template |
|----|-------------------|
| `core_algorithm` | "What is the core algorithm/technique and its key steps?" |
| `datasets` | "What datasets or toy data are relevant, and what preprocessing is needed?" |
| `baselines` | "What baselines or alternative approaches should the demo compare against?" |
| `implementation_details` | "What implementation details, hyperparameters, or tricks matter for a faithful demo?" |
| `evaluation` | "How is the method evaluated — metrics, acceptance checks, or toy assertions?" |

Each direction becomes `Direction{id, question: str, rationale: str}`. The `question` is pre-filled from the template + `Requirements.goal` (LLM only writes `rationale`, not the taxonomy).

* **Fan-out:** `asyncio.gather` over `subagent.run(direction)` — parallel, bounded to 5 concurrent tasks.
* **No re-planning loop yet** — validator may request at most one re-search (YAGNI until measured).

### 4.3 Subagents — `backend/app/rd/subagent.py`

One function, N concurrent invocations. Each subagent:

1. Calls `tools/rag.py:rag_search(direction.question, k=4)` — same hybrid retrieval as `/query`.
2. Synthesizes `ResearchSummary{direction_id, findings: str, sources: list[source_items], confidence: low|med|high, gaps: list[str]}` via a single LLM call (cite sources as `[1]`).

* **Ponytail:** no separate agent processes — `subagent.py` is a plain async function, not a framework. Figures ride along as `object_key` refs (text-only DeepSeek).
* **Events:** each subagent yields `subagent/start` at call time and `subagent/result` on completion so the UI streams progress.

### 4.4 Validator — `backend/app/rd/validator.py`

LLM-as-judge comparing `Requirements` vs. `list[ResearchSummary]`.

* **Output:** `Validation{coverage: dict[direction_id->bool], missing: list[str], feasible: bool, risks: list[str]}`.
* **Behavior:** single LLM call with prompt "check coverage, flag unsupported demo claims, list gaps". If `feasible=false`, **still emit** the artifact with `risks` — never block (`emit-with-risks`). At most one re-search iteration if a gap is trivially fillable with an extra `rag_search`.
* **Ponytail:** single model (`chat_model`), no separate critic model until quality demands it.

### 4.5 Research Artifact — `backend/app/rd/artifacts.py`

Handoff to the coding module. Built from `Requirements + summaries + Validation`:

```python
class DemoSpec(BaseModel):
    entrypoint: str              # "demo.py"
    tech_stack: list[str]        # ["python", "numpy", "gradio?"] — v1: python-only
    core_algorithm: str          # concise steps
    pseudocode: str              # language-agnostic
    acceptance_criteria: list[str]  # runnable checks: ["demo.py exits 0", "toy metric > 0"]

class ResearchArtifact(BaseModel):
    requirements: Requirements
    summaries: list[ResearchSummary]
    validation: Validation
    demo_spec: DemoSpec
    sources: list[dict]          # deduped source_items across subagents
    created_at: float
```

* **Sources:** deduped by `source + page + content` across all summaries.
* **Acceptance criteria** are the contract the coding module must satisfy (the demo's test plan).

### 4.6 Top-level pipeline — `backend/app/rd/pipeline.py`

```python
async def rd_research(requirements: Requirements, session_id: str | None, idea: str | None):
    yield {"type":"orchestrator","event":"plan","directions":[...] }
    summaries = await gather(subagent.run(d) for d in directions)  # streams subagent/* as they complete
    validation = await validator.validate(requirements, summaries)
    artifact = build_artifact(requirements, summaries, validation)
    await db.research_run_upsert(...)  # persist run + artifact
    yield {"type":"validator","event":"result","validation":...}
    yield {"type":"artifact","data":artifact.model_dump()}
```

### 4.7 Tool registry (RAG packaging)

| Module | Function | Used by |
|--------|----------|---------|
| `backend/app/tools/rag.py:rag_search` | `search_kb(question, k, k_images)` | `/query`, `agent.py:kb_search`, `rd/subagent.py` |

`get_figure` remains cut (YAGNI) — figures are `object_key` refs until a vision model is wired (`chat_supports_images=true`).

### 4.8 Streaming protocol (NDJSON, consistent with `/research`)

```
{"type":"orchestrator","event":"plan","directions":[...]}
{"type":"subagent","event":"start","direction_id":"core_algorithm","question":"..."}
{"type":"subagent","event":"result","summary":{...}}
... (x5, parallel — order reflects completion)
{"type":"validator","event":"result","validation":{...}}
{"type":"artifact","data":{...}}
```

Legacy `/research` events (`memory/session`, `agent/tool_call`, etc.) remain unchanged.

### 4.9 Model notes

- `agent_model` (default `deepseek-v4-flash`) supports OpenAI-format tool calling. `deepseek-v4-pro` available for harder planning.
- DeepSeek is text-only: figures are `object_key`/URL refs, never embedded into the model context unless `chat_supports_images` is later set with a vision model.

---

## 5. Memory layer design

### 5.1 Conversation memory (short-term, per session)

- Implemented in `backend/app/memory.py` (`conv_add` / `conv_history`).
- Key: `conv:{session_id}` in Redis; bounded JSON list (last 20 messages), TTL `memory_ttl_s` (24h).
- Appended automatically by `/research` and `/rd/research` (user + assistant turns); loaded into the agent loop's context each call.
- Explicit `/conversations` endpoints are **not** built yet — the UI keeps the `session_id` returned in the first `memory/session` event. Add them only if a conversation-list UI is requested.

### 5.2 Episodic memory (long-term, persistent)

- Implemented in `backend/app/memory.py` (`remember_save` / `remember_search`) + `vectorstore.upsert_episodes` / `search_episodes`.
- Qdrant collection **`episodes`** (bge-m3, dim `embedding_dim`); deterministic point id from payload → idempotent saves.
- Payload: `session_id, created_at, question, answer, sources[]`.
- Saved **automatically** at the end of `/research` (no `memory_save` tool the model must remember to call).
- Retrieval: at the start of `/research`, `remember_search` embeds the question and injects top-`memory_top_k` past episodes into the system context so the orchestrator plans with prior findings in mind.
- UI: no Memory tab yet — the sources/session events already surface memory; add a browse/search tab when requested.

### 5.3 Data flow (legacy /research)

1. `/research` with `session_id` (or a fresh one) loads conversation history + top episodic memories into the orchestrator's context.
2. Orchestrator calls `kb_search` one or more times, streaming `sources` events.
3. On completion: conversation messages are appended, and an episode is saved (`remember_save`) — all automatic, no memory tools for the model.

### 5.4 Data flow (R&D pipeline)

1. Client `POST /rd/analyze {idea, answers?}` → `Requirements` or `questions` (second-request handshake, no KB).
2. Client `POST /rd/research {requirements, session_id}` → orchestrator plans 5 directions, fans out to 5 parallel subagents (each `rag_search` + synthesis), validator checks coverage, artifact is built and persisted to `research_runs` / `research_artifacts`. Conversation + episodic memory are **not** auto-injected in the R&D pipeline (research is artifact-centric, not chat-centric) — add only if cross-run memory proves useful.

---

## 6. Data model additions

- Qdrant collection `episodes` (see §5.2) — implemented.
- Redis keys: `conv:{session_id}` (implemented), `doc:{document_id}` (existing). `research:{research_id}` event-log only if the async job variant is added.
- **New tables:** `research_runs` / `research_artifacts` (see §2.3b). Payload JSON conventions: keep keys flat and snake_case; store `object_key` (not full URLs) for figures.
- Payload JSON conventions: keep keys flat and snake_case; store `object_key` (not full URLs) for figures.

---

## 7. API additions

Implemented:

- `POST /research` — `{question, session_id?, k, k_images}` → NDJSON agent events + final answer; auto-saves an episode and appends conversation.
- `POST /rd/analyze` — `{idea, session_id?, answers?: string[]}` → `{ready, requirements?, questions?}` (second-request handshake, analyzer has no KB).
- `POST /rd/research` — `{requirements, session_id?, idea?}` → NDJSON `orchestrator/plan`, `subagent/*` (x5, parallel), `validator/result`, `artifact`. Fixed 5-direction taxonomy, validator emit-with-risks.

Not built yet (YAGNI until requested):

- `/conversations` CRUD, `/memory/search`, `/research-async` + event replay — only when a conversation-list UI, a Memory tab, or >60s research topics actually need them.
- `/rd/runs` listing — add when run history UI is wanted.

---

## 8. Config additions (`PRODRAG_` prefix) — implemented

```
agent_model=deepseek-v4-flash       # PRODRAG_AGENT_MODEL
agent_max_steps=12
memory_collection=episodes
memory_top_k=5
memory_ttl_s=86400                  # conversation memory TTL
retrieval_hybrid_k=12               # dense+BM25 first-pass size (RRF)
rerank_service_url=""               # optional cross-encoder reranker (Modal)
rerank_service_token=""
rerank_model=BAAI/bge-reranker-base
# R&D pipeline reuses agent_model and retrieval_hybrid_k; no new env vars in v1
```

---

## 9. Frontend (Streamlit, `frontend/`)

Implemented:

- **Research tab** — question input, `k`/`k_images` sliders, live transcript of agent events (`tool_call` / `tool_result` / `sources` figures+text), final answer, session tracking via the `memory/session` event.
- **R&D tab** — idea input, analyzer clarifying questions (second-request handshake, inline form), live subagent progress (5 directions, streaming `subagent/result`), validator risks, and an expandable artifact view (`demo_spec`, `acceptance_criteria`, sources). Session tracking via `session_id` echo.
- **Documents / Ask tabs** — unchanged.

Not built yet (YAGNI until requested):

- **Memory tab** (browse/search past episodes) and a session picker — only when browsing memory is actually wanted.
- **Runs history** for R&D artifacts — only when artifact browsing is wanted.

`frontend/api.py` gained `rd_analyze()` and `rd_research_stream()`.

---

## 10. Conventions & constraints

- **No torch in api/worker images** — all GPU work on Modal.
- **Reuse existing infra** (Qdrant, Redis, Supabase Storage, DeepSeek, Modal). No new services (no separate vector DB, no Postgres) unless justified.
- Keep streaming responses as NDJSON everywhere interactive.
- Deterministic point IDs → idempotent upserts.
- Small tests for non-trivial logic (pytest; no framework ceremony).
- YAGNI: single-loop multi-role agent first; only add multi-process topologies, LangGraph, web search, or a research job queue when measured need exists. R&D subagents are `asyncio.gather`, not a framework.
- Secrets live in `.env`/`.env.example` (placeholders) or Modal secrets; never commit real keys. `.env` is gitignored.
- `uv sync --extra dev --extra modal --extra frontend` (all three).

---

## 11. Roadmap

- **Phase 1 — agent loop (DONE).** `POST /research` streams `kb_search` tool calls → sources → answer; conversation + episodic memory auto-wired. Frontend Research tab. Tests: `test_agent.py`.
- **Phase 2 — R&D research pipeline (THIS PHASE).** `POST /rd/analyze` (second-request handshake, no KB) + `POST /rd/research` (fixed 5-direction fan-out, parallel subagents, validator emit-with-risks, `ResearchArtifact` handoff). Frontend R&D tab. Tests: `test_rd_*.py`.
- **Phase 3 — coding module.** Consumes `ResearchArtifact` and emits `demo.py + requirements.txt` (v1) with runnable `acceptance_criteria` checks.
- **Phase 4 — conversation UX + Memory tab.** Session picker / "new session" in the UI; `/conversations` endpoints only if a conversation list is wanted.
- **Phase 5 — critic + web search (optional).** draft review for unsupported claims; `web_search` tool if requested.
- **Phase 6 — eval for agentic quality.** extend `eval/` beyond recall@k: answer faithfulness (citation coverage) + R&D artifact faithfulness.

---

## 12. Running & testing (current)

```bash
# deps (ALL extras)
uv sync --extra dev --extra modal --extra frontend

# infra (needs Supabase creds in .env first, see §2.7)
docker compose -f deploy/docker-compose.yml up --build

# apply schema (includes research_runs / research_artifacts)
psql "$PRODRAG_DATABASE_URL" -f backend/app/core/schema.sql
# or paste backend/app/core/schema.sql into Supabase SQL editor

# tests
uv run pytest

# frontend
cd frontend && uv run streamlit run app.py

# modal service lifecycle
bash deploy/modal/manage.sh {status|stop|start}
```

Blocked until the user supplies `PRODRAG_SUPABASE_URL` / `PRODRAG_SUPABASE_SECRET_KEY` and creates the `prodrag-assets` bucket.
