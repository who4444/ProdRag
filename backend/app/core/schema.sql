-- ProdRag database schema (Supabase Postgres / public schema).
-- Apply via Supabase SQL editor (or `psql` with a DATABASE_URL).

create table if not exists public.documents (
  id            uuid primary key,
  title         text not null,
  source_key    text not null,
  content_hash  text,
  metadata      jsonb not null default '{}'::jsonb,
  status        text not null default 'queued',
  text_chunks   int  not null default 0,
  images        int  not null default 0,
  error         text,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

create index if not exists documents_status_idx on public.documents (status);

create table if not exists public.parts (
  id            uuid primary key,
  document_id   uuid not null references public.documents(id) on delete cascade,
  kind          text not null,
  page          int  not null,
  block_order   int,
  caption       text,
  ocr_text      text,
  object_key    text
);

create index if not exists parts_document_id_idx on public.parts (document_id);

create table if not exists public.conversations (
  id          uuid primary key,
  created_at  timestamptz not null default now()
);

create table if not exists public.messages (
  id              bigserial primary key,
  conversation_id uuid not null references public.conversations(id) on delete cascade,
  role            text not null,
  content         text not null,
  created_at      timestamptz not null default now()
);

create index if not exists messages_conversation_id_idx
  on public.messages (conversation_id, id);

create table if not exists public.episodes (
  id          uuid primary key,
  session_id  uuid,
  question    text not null,
  answer      text not null,
  sources     jsonb not null default '[]'::jsonb,
  created_at  timestamptz not null default now()
);

-- R&D pipeline — research runs and materialized artifacts (handoff to coding module)
create table if not exists public.research_runs (
  id            uuid primary key,
  session_id    uuid,
  idea          text not null,
  requirements  jsonb not null default '{}'::jsonb,
  status        text not null default 'running',
  created_at    timestamptz not null default now()
);

create table if not exists public.research_artifacts (
  run_id        uuid primary key references public.research_runs(id) on delete cascade,
  artifact      jsonb not null,
  validation    jsonb not null default '{}'::jsonb,
  created_at    timestamptz not null default now()
);

-- Code generation — code runs and artifacts (demo.py + tests + sandbox)
create table if not exists public.code_runs (
  id            uuid primary key,
  artifact_run_id uuid,
  coding_spec   jsonb not null default '{}'::jsonb,
  files         jsonb not null default '[]'::jsonb,
  status        text not null default 'running',
  created_at    timestamptz not null default now()
);

create table if not exists public.code_artifacts (
  run_id        uuid primary key references public.code_runs(id) on delete cascade,
  artifact      jsonb not null,
  sandbox       jsonb not null default '{}'::jsonb,
  created_at    timestamptz not null default now()
);
