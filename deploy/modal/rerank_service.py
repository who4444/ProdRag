"""Modal GPU service: cross-encoder reranker for ProdRag.

BAAI/bge-reranker-base on a T4 GPU, called by the CPU-side API/worker over
HTTPS. Scores (query, passage) pairs; higher is more relevant.

Deploy:
    modal secret create prodrag-rerank-token AUTH_TOKEN=<random>   # once
    modal deploy deploy/modal/rerank_service.py

The deploy prints the service URL, e.g.
https://<workspace>--prodrag-rerank-reranker-rerank.modal.run
Set PRODRAG_RERANK_SERVICE_URL=<that URL> and
PRODRAG_RERANK_SERVICE_TOKEN=<AUTH_TOKEN>. Retrieval falls back to RRF order
when the URL is unset.
"""

import os

import fastapi
import modal
from pydantic import BaseModel

MODEL_NAME = "BAAI/bge-reranker-base"

rerank_image = (
    modal.Image.debian_slim().pip_install(
        "fastapi[standard]",
        "pydantic",
        "sentence-transformers>=3.0",
    )
)

app = modal.App("prodrag-rerank", image=rerank_image)


class RerankRequest(BaseModel):
    query: str
    texts: list[str]


class RerankResponse(BaseModel):
    scores: list[float]
    model: str = MODEL_NAME


@app.cls(
    gpu="T4",
    secrets=[modal.Secret.from_name("prodrag-rerank-token", required_keys=["AUTH_TOKEN"])],
    min_containers=0,
    max_containers=2,
    timeout=300,
)
@modal.concurrent(max_inputs=4)
class Reranker:
    @modal.enter()
    def load(self):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(MODEL_NAME, device="cuda", max_length=512)

    def _check_auth(self, request: fastapi.Request) -> None:
        expected = os.environ.get("AUTH_TOKEN", "")
        if not expected:
            return
        if request.headers.get("Authorization") != f"Bearer {expected}":
            raise fastapi.HTTPException(status_code=401, detail="unauthorized")

    @modal.fastapi_endpoint(method="POST")
    def rerank(self, request: fastapi.Request, body: RerankRequest) -> RerankResponse:
        self._check_auth(request)
        scores = self.model.predict([[body.query, t] for t in body.texts]).tolist()
        return RerankResponse(scores=scores)