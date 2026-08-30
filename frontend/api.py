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
