# ProdRag — R&D Demo System (v1)

Turns research papers and ideas into **runnable demo artifacts** — `demo.py + requirements.txt + test_demo.py` that demonstrates a paper's core idea on toy data, tested in an isolated sandbox.

> **Pipeline:** `idea -> research -> code`. The multimodal RAG stack is packaged as a reusable tool for both chat and the R&D pipelines. See **[PROJECT.md](PROJECT.md)** for the living spec, data model, and roadmap.

## Architecture (v1)

```
                         ┌──────────────────────────────────────────────────┐
                         │                 FastAPI (api)                     │
                         │  /rd/analyze  /rd/research  /code/*  /query      │
                         │  /research  /documents  /files  /tools/search    │
                         │  /health                                          │
                         └───────┬──────────────────────┬───────────────────┘
                                 │                      │
               ┌─────────────────┘                      └─────────────────┐
               │ R&D Pipeline (idea → artifact)          │  Code Pipeline (artifact → demo)
               │ Analyzer (no KB, 2-req handshake)        │  Analyzer (artifact → CodingSpec)
               │ Orchestrator (fixed 5 dirs)               │  Orchestrator (fixed 3 files)
               │ Subagents x5 (rag_search + synth)          │  Subagents x3 (one LLM / file)
               │ Validator (emit-with-risks)                │  Tester (from spec, not code)
               │ Artifact (demo_spec + criteria)           │  Sandbox (tmpdir+subprocess)
               └──────────┬────────────────────────────────┴──────────┬───────┘
                          │  RAG Tool (tools/rag.py)                  │
                          │  search_kb() hybrid dense+BM25+rerank+CLIP│
                          │  web_search (tools/search.py) Tavily/Serper│
                          ▼                                           ▼
                 Supabase Postgres                              Qdrant / Redis / Storage
                 (documents, parts, research_runs,               (text_chunks, image_chunks,
                  research_artifacts, code_runs,                  episodes)
                  code_artifacts, conv)                          Supabase Storage (PDFs, figures)
                                                                 Redis (queue, conv:{session})
                                                                 Modal GPU (bge-m3, CLIP) + DeepSeek
```

* **RAG Tool** (`backend/app/tools/rag.py`): `search_kb()` hybrid dense+BM25 via RRF + rerank + CLIP. Shared by `/query`, legacy `/research`, R&D subagents.
* **Search Tool** (`backend/app/tools/search.py`): `web_search()` via Tavily/Serper/generic proxy; `enabled()` false → no-op. R&D subagents auto-append web hits when RAG `<2`; legacy agent exposes `web_search` as second tool when configured.
* **R&D Subagents** (`backend/app/rd/subagent.py`): `rag_search` + web fallback → `ResearchSummary{findings, sources, confidence, gaps}`.
* **Code Subagents** (`backend/app/code/subagent.py`): one LLM call per `FileTask` → `GeneratedFile{path, content}`.

v1 demo is minimal: `demo.py` (toy data, `python`/`numpy`), `requirements.txt`, `test_demo.py` — sandbox-verified. Gradio/Docker are v2.

## Data Flow (v1)

```
User idea (free text)
  ──POST /rd/analyze {idea}────────────────────────► Analyzer
        │  {ready:false, questions:[...]}  ◄────────┘ (no KB, max 2 rounds)
        │  {ready:true, requirements:{goal, core_idea, demo_type, constraints, audience}}
        ▼
  POST /rd/research {requirements, session_id, idea?}
        │  NDJSON stream:
        │    orchestrator/plan (5 dirs: core_algorithm, datasets, baselines, implementation_details, evaluation)
        │    subagent/start  ─┐
        │    subagent/result ─┤  x5 (each: rag_search + web fallback → LLM synth)
        │    validator/result ─┤  LLM-as-judge {coverage, missing, feasible, risks} — emit-with-risks
        │    artifact          ─┘  ResearchArtifact{demo_spec{entrypoint, tech_stack, core_algorithm, pseudocode, acceptance_criteria}, sources deduped}
        │  Persist: research_runs + research_artifacts (Supabase Postgres)  [best-effort, never fails stream]
        ▼
  POST /code/analyze {artifact}
        │  → CodingSpec{file_tasks:[demo.py, requirements.txt, test_demo.py], test_intents, dependencies, risks}
        │     One json_object LLM call, fallback if LLM down, no KB
        ▼
  POST /code/generate {artifact, session_id?}
        │  NDJSON stream:
        │    code/plan {coding_spec}
        │    code/file_start|file_result x3  (one LLM / file, markdown-fence stripped)
        │    tester/result {test_demo.py}  (from CodingSpec.test_intents + acceptance_criteria, not from demo.py)
        │    sandbox/start
        │    sandbox/result {passed, pytest_log, demo_log, returncode, runtime_ms}  (tmpdir + subprocess, pip 60s, pytest/demo 30s)
        │    code/artifact {CodeArtifact{files, test_file, sandbox}}
        │  Persist: code_runs + code_artifacts  [best-effort]
        ▼
  CodeArtifact — handoff to user / next stage. Files are runnable: pip install -r requirements.txt && pytest && python demo.py
```

If idea is precise, first `POST /rd/analyze` returns `ready:true` immediately. `ResearchArtifact` may be `feasible:false` — pipeline still emits (risks propagated to `CodeArtifact`).

Ingestion (async): `POST /documents` → SHA-256 dedup → Supabase Storage → `arq` job `ingest_document` (`backend/app/ingest/pipeline.py`) → PyMuPDF parse → chunk → bge-m3 (Modal) + CLIP (Modal) → Qdrant (`text_chunks` hybrid + `image_chunks`) → `parts` rows. Re-ingest cleans orphans (`delete_points` + `delete_object`).

## API (v1)

All except `GET /health` require `Authorization: Bearer $PRODRAG_API_TOKEN`.

| Endpoint | Method | Request | Response | Notes |
|---|---|---|---|---|
| `GET /health` | GET | — | `{"status":"ok"}` | unauthenticated |
| `POST /documents` | POST | `multipart file + metadata` | `{"document_id","job_id"}` or `{"document_id", duplicate:true}` | SHA-256 dedup |
| `POST /documents/{id}/reingest` | POST | — | `{"document_id","job_id"}` | cleans old points/figures |
| `GET /documents/{id}/status` | GET | — | `{"status","text_chunks","images","error"}` | |
| `GET /documents` / `DELETE /documents/{id}` | GET/DELETE | — | list / `{"deleted":id}` | |
| `POST /query` | POST | `{"question", k, k_images}` | NDJSON `sources` then `text` deltas | RAG answer |
| `POST /research` | POST | `{"question","session_id?", k, k_images}` | NDJSON `memory/session`, `agent/tool_call`, `sources`, `text`, `memory/saved` | legacy single-loop; `web_search` added when configured |
| `POST /rd/analyze` | POST | `{"idea","session_id?","answers?":string[]}` | `{"ready", "requirements?","questions?","draft?"}` | second-request handshake, no KB |
| `POST /rd/research` | POST | `{"requirements","session_id?","idea?", k?, k_images?}` | NDJSON `orchestrator/plan`, `subagent/*` x5, `validator/result`, `artifact` | fixed taxonomy, emit-with-risks |
| `POST /code/analyze` | POST | `{"artifact": ResearchArtifact}` | `{"file_tasks","test_intents","dependencies","risks", "coding_spec":{...}}` | one json_object call, fallback on failure |
| `POST /code/generate` | POST | `{"artifact": ResearchArtifact, "session_id?"}` | NDJSON `code/plan`, `code/file_start|file_result` x3, `tester/result`, `sandbox/start|result`, `code/artifact` | `code/artifact` contains `CodeArtifact{files, test_file, sandbox, coding_spec}` |
| `POST /tools/search` | POST | `{"query", k?}` | `{"enabled","results":[{title,url,snippet}],"formatted"}` | web_search (Tavily/Serper/generic), `[]` when unconfigured |
| `GET /tools/search` | GET | `?query=&k=` | same | probe from browser (avoids 405) |
| `GET /tools/search/status` | GET | — | `{"enabled":bool}` | |
| `GET /files/{object_key}` | GET | — | bytes | proxies Supabase Storage |

NDJSON everywhere streaming (`application/x-ndjson`, each line `JSON + "\n"`).

**Example — full pipeline (curl):**

```bash
TOKEN=change-me; API=http://localhost:8000

# 1. Analyze (may ask clarifying Qs)
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"idea":"Turn Attention Is All You Need into a minimal demo"}' $API/rd/analyze | jq

# 1b. If ready:false, re-POST with answers
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"idea":"Turn Attention Is All You Need into a minimal demo","answers":["toy data","python script"]}' $API/rd/analyze | jq

# 2. Research (stream)
curl -N -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"requirements":{"goal":"demo attention","core_idea":"attention","demo_type":"script","constraints":[],"audience":"researcher","open_questions":[]},"idea":"demo attention"}' $API/rd/research

# Save artifact from last line: {"type":"artifact","data":{...}}
# 3. Code analyze
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"artifact":'"$(cat artifact.json)"'}' $API/code/analyze | jq

# 4. Code generate (stream)
curl -N -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"artifact":'"$(cat artifact.json)"'}' $API/code/generate

# Web search probe
curl -s -H "Authorization: Bearer $TOKEN" "http://localhost:8000/tools/search?query=transformer+2024&k=3" | jq
```

## Data Model (Supabase Postgres, `backend/app/core/schema.sql`)

| Table | Key | Purpose |
|---|---|---|
| `documents` | `id uuid` | lifecycle `queued/processing/done/error`, `source_key`, `content_hash`, `text_chunks`, `images` |
| `parts` | `id uuid` | one row per Qdrant point (`id` = point id `kind+payload` SHA-256), `document_id fk`, `kind`, `page`, `object_key` |
| `conversations` / `messages` | `id uuid` / `bigserial` | durable chat (`conv:{session}` in Redis is cache) |
| `episodes` | `id uuid` | episodic memory metadata (`vector` in Qdrant `episodes`) |
| `research_runs` | `id uuid` | `session_id?`, `idea`, `requirements jsonb`, `status` |
| `research_artifacts` | `run_id uuid` | `artifact jsonb` (ResearchArtifact), `validation jsonb` |
| `code_runs` | `id uuid` | `artifact_run_id?`, `coding_spec jsonb`, `files jsonb`, `status` |
| `code_artifacts` | `run_id uuid` | `artifact jsonb` (CodeArtifact), `sandbox jsonb` |

Qdrant collections: `text_chunks` (bge-m3 1024 + BM25 sparse, RRF `Fusion.RRF`), `image_chunks` (CLIP 512), `episodes` (bge-m3 1024). Point IDs deterministic (`backend/app/core/vectorstore.py:82`).

## Run

```bash
cp .env.example .env   # set PRODRAG_SUPABASE_URL/_SECRET_KEY, PRODRAG_CHAT_API_KEY, PRODRAG_API_TOKEN
# optional: PRODRAG_TAVILY_API_KEY or PRODRAG_SEARCH_SERVICE_URL for web_search
docker compose -f deploy/docker-compose.yml up --build
curl localhost:8000/health

# apply schema (includes code_runs/code_artifacts)
psql "$PRODRAG_DATABASE_URL" -f backend/app/core/schema.sql
# or paste into Supabase SQL editor
```

Env: `PRODRAG_` prefix (`backend/app/config.py:6` reads `../.env` in Docker, `backend/.env` locally). See `.env.example` for all vars including `PRODRAG_SEARCH_*`.

## Local dev / tests

```bash
uv sync --extra dev
uv run pytest -q   # 137 tests (RD + code + search, pytest-asyncio)
uv run uvicorn app.main:app --reload   # from backend/, needs local qdrant/redis + Supabase
```

## Frontend

```bash
uv sync --extra frontend
cd frontend && uv run streamlit run app.py   # 4 tabs: Documents / Ask / Research / R&D
# Simple test UIs (no Streamlit):
open frontend/simple_test.html          # pure HTML/JS — health, docs, RAG, R&D, web search
# or
cd frontend && uv run streamlit run simple_test_app.py
```

`frontend/api.py` thin `requests` client; `frontend/simple_test.html` uses `fetch` + NDJSON `reader.getReader()` loop.

## GPU services on Modal

```bash
uv sync --extra dev --extra modal
modal secret create prodrag-clip-token AUTH_TOKEN="$(openssl rand -hex 24)"   # once
modal secret create prodrag-embed-token AUTH_TOKEN="$(openssl rand -hex 24)"   # once
modal deploy deploy/modal/embed_service.py
modal deploy deploy/modal/clip_service.py
# set PRODRAG_EMBED_SERVICE_URL/_TOKEN and PRODRAG_CLIP_SERVICE_URL/_TOKEN in .env
```

`deploy/modal/search_proxy.py` example for generic proxy → `PRODRAG_SEARCH_SERVICE_URL`. Cold start ~30–60s; client timeouts 15s for search, 120s for embeddings. `scale_to_zero` (`min_containers=0`).

## Notes

* Modal URLs absent → fallback embeddings; `PRODRAG_EMBEDDING_DIM` must match collection.
* `uv sync --extra dev --extra modal --extra frontend` (all three).
* `ensure_collections()` swallows 409 — safe for concurrent api+worker boot.
* Point IDs deterministic → idempotent re-ingestion; re-ingest cleans orphans.
* Search `enabled()` false → RAG-only, never raises (`backend/app/tools/search.py:19`); R&D subagents only call `web_search` when RAG `<2`.
* CORS `allow_origins=["*"]` in `backend/app/main.py` for `file://` test UI; restrict in prod.
* Single `PRODRAG_API_TOKEN` Bearer auth (`backend/app/api/__init__.py:8`).
```

