"""Code generation endpoints: /code/analyze + /code/generate."""

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..code.analyzer import analyze_artifact
from ..code.pipeline import code_generate

router = APIRouter(prefix="/code")


def _extract_artifact(body: dict) -> dict | None:
    if not isinstance(body, dict) or not body:
        return None
    # explicit artifact key
    if "artifact" in body:
        art = body.get("artifact")
        if isinstance(art, dict) and art:
            return art
        return None
    # body itself is the artifact (has requirements/demo_spec etc)
    if any(k in body for k in ("requirements", "demo_spec", "summaries", "validation")):
        return body
    return None


@router.post("/analyze")
async def code_analyze(request: Request):
    body = await request.json()
    artifact = _extract_artifact(body)
    if artifact is None:
        raise HTTPException(status_code=422, detail="artifact required")
    coding_spec = await analyze_artifact(request.app.state.client, artifact)
    data = coding_spec.model_dump()
    return JSONResponse({**data, "coding_spec": data})


@router.post("/generate")
async def code_generate_endpoint(request: Request):
    body = await request.json()
    artifact = _extract_artifact(body)
    if artifact is None:
        raise HTTPException(status_code=422, detail="artifact required")
    session_id = body.get("session_id")

    async def _gen():
        async for evt in code_generate(request.app.state.client, artifact, session_id=session_id):
            yield json.dumps(evt) + "\n"

    return StreamingResponse(_gen(), media_type="application/x-ndjson")
