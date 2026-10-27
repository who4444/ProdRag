"""Thin client for the ProdRag HTTP API."""

import json

import requests


class APIError(RuntimeError):
    pass


def _check(resp: requests.Response) -> None:
    if resp.status_code >= 400:
        raise APIError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    resp.raise_for_status()


def health(base_url: str, timeout: float = 5.0) -> dict:
    r = requests.get(f"{base_url}/health", timeout=timeout)
    _check(r)
    return r.json()


def upload_document(
    base_url: str, token: str, filename: str, data: bytes, metadata: dict | None = None
) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(
        f"{base_url}/documents",
        headers=headers,
        files={"file": (filename, data, "application/pdf")},
        data={"metadata": json.dumps(metadata or {})},
        timeout=120,
    )
    _check(r)
    return r.json()


def document_status(base_url: str, token: str, doc_id: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{base_url}/documents/{doc_id}/status", headers=headers, timeout=30)
    _check(r)
    return r.json()


def query_stream(base_url: str, token: str, question: str, k: int, k_images: int):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"question": question, "k": k, "k_images": k_images}
    r = requests.post(
        f"{base_url}/query", headers=headers, json=payload, stream=True, timeout=300
    )
    if r.status_code >= 400:
        raise APIError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r


def research_stream(
    base_url: str,
    token: str,
    question: str,
    session_id: str | None = None,
    k: int = 4,
    k_images: int = 1,
):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "question": question,
        "session_id": session_id,
        "k": k,
        "k_images": k_images,
    }
    r = requests.post(
        f"{base_url}/research", headers=headers, json=payload, stream=True, timeout=300
    )
    if r.status_code >= 400:
        raise APIError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r


def get_file(base_url: str, token: str, object_key: str) -> bytes:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{base_url}/files/{object_key}", headers=headers, timeout=60)
    _check(r)
    return r.content


def rd_analyze(base_url: str, token: str, idea: str, session_id: str | None = None, answers: list[str] | None = None) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    payload: dict = {"idea": idea}
    if session_id:
        payload["session_id"] = session_id
    if answers:
        payload["answers"] = answers
    r = requests.post(f"{base_url}/rd/analyze", headers=headers, json=payload, timeout=60)
    _check(r)
    return r.json()


def rd_research_stream(base_url: str, token: str, requirements: dict, session_id: str | None = None, idea: str | None = None, k: int = 4, k_images: int = 1):
    headers = {"Authorization": f"Bearer {token}"}
    payload: dict = {"requirements": requirements, "k": k, "k_images": k_images}
    if session_id:
        payload["session_id"] = session_id
    if idea:
        payload["idea"] = idea
    r = requests.post(f"{base_url}/rd/research", headers=headers, json=payload, stream=True, timeout=300)
    if r.status_code >= 400:
        raise APIError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r


def code_analyze(base_url: str, token: str, artifact: dict) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(f"{base_url}/code/analyze", headers=headers, json={"artifact": artifact}, timeout=60)
    _check(r)
    return r.json()


def code_generate_stream(base_url: str, token: str, artifact: dict, session_id: str | None = None):
    headers = {"Authorization": f"Bearer {token}"}
    payload: dict = {"artifact": artifact}
    if session_id:
        payload["session_id"] = session_id
    r = requests.post(f"{base_url}/code/generate", headers=headers, json=payload, stream=True, timeout=300)
    if r.status_code >= 400:
        raise APIError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r


def repo_index(base_url: str, token: str, path: str, repo_id: str | None = None, name: str | None = None) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    payload: dict = {"path": path}
    if repo_id:
        payload["repo_id"] = repo_id
    if name:
        payload["name"] = name
    r = requests.post(f"{base_url}/repo/index", headers=headers, json=payload, timeout=600)
    _check(r)
    return r.json()


def repo_list(base_url: str, token: str) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{base_url}/repo/list", headers=headers, timeout=30)
    _check(r)
    return r.json()


def repo_search(base_url: str, token: str, query: str, repo_id: str | None = None, k: int = 6) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    payload: dict = {"query": query, "k": k}
    if repo_id:
        payload["repo_id"] = repo_id
    r = requests.post(f"{base_url}/repo/search", headers=headers, json=payload, timeout=120)
    _check(r)
    return r.json()


def code_extend_stream(
    base_url: str,
    token: str,
    idea: str,
    repo_id: str | None = None,
    path: str | None = None,
    entrypoint: str | None = None,
    session_id: str | None = None,
):
    headers = {"Authorization": f"Bearer {token}"}
    payload: dict = {"idea": idea}
    if repo_id:
        payload["repo_id"] = repo_id
    if path:
        payload["path"] = path
    if entrypoint:
        payload["entrypoint"] = entrypoint
    if session_id:
        payload["session_id"] = session_id
    r = requests.post(f"{base_url}/code/extend", headers=headers, json=payload, stream=True, timeout=600)
    if r.status_code >= 400:
        raise APIError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r


def feasibility_understand(
    base_url: str, token: str, idea: str, answers: list[str] | None = None, session_id: str | None = None
) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    payload: dict = {"idea": idea}
    if answers:
        payload["answers"] = answers
    if session_id:
        payload["session_id"] = session_id
    r = requests.post(f"{base_url}/feasibility/understand", headers=headers, json=payload, timeout=60)
    _check(r)
    return r.json()


def feasibility_stream(
    base_url: str,
    token: str,
    idea: str,
    repo_id: str,
    understanding: dict | None = None,
    answers: list[str] | None = None,
    k: int = 6,
):
    headers = {"Authorization": f"Bearer {token}"}
    payload: dict = {"idea": idea, "repo_id": repo_id, "k": k}
    if understanding:
        payload["understanding"] = understanding
    if answers:
        payload["answers"] = answers
    r = requests.post(
        f"{base_url}/feasibility/analyze", headers=headers, json=payload, stream=True, timeout=600
    )
    if r.status_code >= 400:
        raise APIError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r


def list_documents(base_url: str, token: str, status: str | None = None) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}
    params = {}
    if status:
        params["status"] = status
    r = requests.get(f"{base_url}/documents", headers=headers, params=params, timeout=30)
    _check(r)
    return r.json()


def document_delete(base_url: str, token: str, doc_id: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.delete(f"{base_url}/documents/{doc_id}", headers=headers, timeout=30)
    _check(r)
    return r.json()


def document_reingest(base_url: str, token: str, doc_id: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(f"{base_url}/documents/{doc_id}/reingest", headers=headers, timeout=30)
    _check(r)
    return r.json()


def list_rd_runs(base_url: str, token: str, limit: int = 20, offset: int = 0, session_id: str | None = None, status: str | None = None) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}
    params: dict = {"limit": limit, "offset": offset}
    if session_id:
        params["session_id"] = session_id
    if status:
        params["status"] = status
    r = requests.get(f"{base_url}/rd/runs", headers=headers, params=params, timeout=30)
    _check(r)
    return r.json()


def get_rd_run(base_url: str, token: str, run_id: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{base_url}/rd/runs/{run_id}", headers=headers, timeout=30)
    _check(r)
    return r.json()


def list_code_runs(base_url: str, token: str, limit: int = 20, offset: int = 0, status: str | None = None) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}
    params: dict = {"limit": limit, "offset": offset}
    if status:
        params["status"] = status
    r = requests.get(f"{base_url}/code/runs", headers=headers, params=params, timeout=30)
    _check(r)
    return r.json()


def get_code_run(base_url: str, token: str, run_id: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{base_url}/code/runs/{run_id}", headers=headers, timeout=30)
    _check(r)
    return r.json()


def list_conversations(base_url: str, token: str, limit: int = 20, offset: int = 0) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{base_url}/conversations", headers=headers, params={"limit": limit, "offset": offset}, timeout=30)
    _check(r)
    return r.json()


def get_conversation_messages(base_url: str, token: str, conv_id: str, n: int = 20) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{base_url}/conversations/{conv_id}/messages", headers=headers, params={"n": n}, timeout=30)
    _check(r)
    return r.json()


def memory_search(base_url: str, token: str, query: str, k: int = 5) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{base_url}/memory/search", headers=headers, params={"query": query, "k": k}, timeout=30)
    _check(r)
    return r.json()


def memory_episodes(base_url: str, token: str, limit: int = 20) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{base_url}/memory/episodes", headers=headers, params={"limit": limit}, timeout=30)
    _check(r)
    return r.json()


def web_search_probe(base_url: str, token: str, query: str, k: int = 5) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(f"{base_url}/tools/search", headers=headers, json={"query": query, "k": k}, timeout=30)
    _check(r)
    return r.json()
