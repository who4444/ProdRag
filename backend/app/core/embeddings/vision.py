"""CLIP embeddings. Remote-first: calls the Modal GPU service
(deploy/modal/clip_service.py) when PRODRAG_CLIP_SERVICE_URL is set.

Falls back to a lazy local open_clip load (requires torch installed) for
offline dev — the backend ships without torch in production.
"""

import asyncio
import base64
import io
import threading

import httpx

from ...config import settings

_model = None
_preprocess = None
_lock = threading.Lock()


def _remote_url() -> str | None:
    url = (settings.clip_service_url or "").strip().rstrip("/")
    return url or None


async def _remote_embed(kind: str, items: list[str]) -> list[list[float]]:
    headers = {}
    if settings.clip_service_token:
        headers["Authorization"] = f"Bearer {settings.clip_service_token}"
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            _remote_url(),
            json={"type": kind, "data": items},
            headers=headers,
        )
        resp.raise_for_status()
    return resp.json()["embeddings"]


async def embed_images(pngs: list[bytes]) -> list[list[float]]:
    if not pngs:
        return []
    if _remote_url():
        return await _remote_embed("image", [base64.b64encode(p).decode() for p in pngs])
    return await asyncio.to_thread(_embed_images_sync, pngs)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if _remote_url():
        return await _remote_embed("text", texts)
    return await asyncio.to_thread(_embed_texts_sync, texts)


def _load() -> None:
    global _model, _preprocess
    with _lock:
        if _model is None:
            import open_clip

            _model, _, _preprocess = open_clip.create_model_and_transforms(
                settings.vision_embedding_model,
                pretrained=settings.vision_embedding_pretrained,
            )
            _model.eval()


def _embed_images_sync(pngs: list[bytes]) -> list[list[float]]:
    import torch
    from PIL import Image

    _load()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _model.to(device)
    tensors = [_preprocess(Image.open(io.BytesIO(p))).to(device) for p in pngs]
    with torch.no_grad():
        feats = model.encode_image(torch.stack(tensors))
    return torch.nn.functional.normalize(feats, dim=-1).cpu().tolist()


def _embed_texts_sync(texts: list[str]) -> list[list[float]]:
    import torch
    import open_clip

    _load()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _model.to(device)
    tokenized = open_clip.tokenize(texts)
    with torch.no_grad():
        feats = model.encode_text(tokenized.to(device))
    return torch.nn.functional.normalize(feats, dim=-1).cpu().tolist()
