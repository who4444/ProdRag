"""Modal GPU service: text embeddings for ProdRag.

Self-hosted BAAI/bge-m3 embeddings on a T4 GPU, called by the CPU-side
API/worker over HTTPS. bge-m3 is a strong multilingual retrieval model
(1024 dims, up to 8k token inputs). Query and passage inputs use the
recommended instruction-prefix convention (queries only).

Deploy:
    modal secret create prodrag-embed-token AUTH_TOKEN=<random>   # once
    modal deploy deploy/modal/embed_service.py

The deploy prints the service URL, e.g.
https://<workspace>--prodrag-embed-textembedder-embed.modal.run
Set PRODRAG_EMBED_SERVICE_URL=<that URL>, PRODRAG_EMBED_SERVICE_TOKEN=<AUTH_TOKEN>,
and PRODRAG_EMBEDDING_DIM=1024 (must match this model).
"""

import os

import fastapi
import modal
from pydantic import BaseModel

MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

embed_image = (
    modal.Image.debian_slim().pip_install(
        "fastapi[standard]",
        "pydantic",
        "sentence-transformers>=3.0",
    )
)

app = modal.App("prodrag-embed", image=embed_image)


class EmbedRequest(BaseModel):
    data: list[str]
    query: bool = False  # apply the retrieval instruction prefix


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str = MODEL_NAME
    dim: int = EMBEDDING_DIM


@app.cls(
    gpu="T4",
    secrets=[modal.Secret.from_name("prodrag-embed-token", required_keys=["AUTH_TOKEN"])],
    min_containers=1,
    max_containers=4,
    timeout=300,
)
@modal.concurrent(max_inputs=8)
class TextEmbedder:
    @modal.enter()
    def load(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME, device="cuda")

    def _check_auth(self, request: fastapi.Request) -> None:
        expected = os.environ.get("AUTH_TOKEN", "")
        if not expected:
            return
        if request.headers.get("Authorization") != f"Bearer {expected}":
            raise fastapi.HTTPException(status_code=401, detail="unauthorized")

    @modal.fastapi_endpoint(method="POST")
    def embed(self, request: fastapi.Request, body: EmbedRequest) -> EmbedResponse:
        self._check_auth(request)
        texts = [QUERY_PREFIX + t if body.query else t for t in body.data]
        vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return EmbedResponse(embeddings=vectors.tolist())
