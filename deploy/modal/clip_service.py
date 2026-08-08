"""Modal GPU service: CLIP vision embeddings for ProdRag.

The backend (FastAPI API + arq worker) runs CLIP-free on CPU and calls this
service over HTTPS to embed figures and query text into CLIP's shared
vision-text space. Kept on GPU so batch inference is fast and the heavy
torch stack never ships with the API/worker images.

Deploy:
    modal secret create prodrag-clip-token AUTH_TOKEN=<random>   # once
    modal deploy deploy/modal/clip_service.py

The deploy prints the service URL, e.g. https://<workspace>--prodrag-clip-clipembedder-embed.modal.run
Set PRODRAG_CLIP_SERVICE_URL=<that URL> and PRODRAG_CLIP_SERVICE_TOKEN=<AUTH_TOKEN>.
"""

import base64
import io
import os

import fastapi
import modal
from pydantic import BaseModel

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

clip_image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi[standard]",
        "pydantic",
        "pillow",
        "torch",
        "torchvision",
        "open-clip-torch>=2.24",
    )
)

app = modal.App("prodrag-clip", image=clip_image)


class EmbedRequest(BaseModel):
    type: str  # "image" | "text"
    data: list[str]  # base64-encoded PNGs for images, raw strings for text


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


@app.cls(
    gpu="T4",
    secrets=[modal.Secret.from_name("prodrag-clip-token", required_keys=["AUTH_TOKEN"])],
    min_containers=1,
    max_containers=4,
    timeout=300,
)
@modal.concurrent(max_inputs=8)
class ClipEmbedder:
    @modal.enter()
    def load(self):
        import open_clip
        import torch

        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED, device="cuda"
        )
        self.model.eval()

    def _check_auth(self, request: fastapi.Request) -> None:
        expected = os.environ.get("AUTH_TOKEN", "")
        if not expected:
            return
        if request.headers.get("Authorization") != f"Bearer {expected}":
            raise fastapi.HTTPException(status_code=401, detail="unauthorized")

    @modal.fastapi_endpoint(method="POST")
    def embed(self, request: fastapi.Request, body: EmbedRequest) -> EmbedResponse:
        self._check_auth(request)
        import open_clip
        from PIL import Image

        torch = self.torch
        with torch.no_grad():
            if body.type == "image":
                images = [
                    self.preprocess(Image.open(io.BytesIO(base64.b64decode(b))))
                    for b in body.data
                ]
                feats = self.model.encode_image(torch.stack(images).to("cuda"))
            elif body.type == "text":
                tokens = open_clip.tokenize(body.data).to("cuda")
                feats = self.model.encode_text(tokens)
            else:
                raise fastapi.HTTPException(status_code=400, detail="type must be image or text")
        normalized = torch.nn.functional.normalize(feats, dim=-1)
        return EmbedResponse(embeddings=normalized.cpu().tolist())
