"""Code Analyzer — ResearchArtifact -> CodingSpec.

One json_object LLM call, no KB, fallback, never raises.
"""

import json
import logging

from ..config import settings
from .schemas import CodingSpec, FileTask

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a code analyzer for an R&D demo system. Given a ResearchArtifact "
    "(requirements, demo_spec, summaries, validation), output a CodingSpec JSON.\n"
    "Shape: {\"file_tasks\": [{\"path\": str, \"goal\": str, \"context_slice\": str}], "
    "\"test_intents\": [str], \"dependencies\": [str], \"risks\": [str]}\n"
    "Rules:\n"
    "- file_tasks must include demo.py, requirements.txt, test_demo.py (v1 fixed set).\n"
    "- test_intents must derive from acceptance_criteria (e.g. demo.py exits 0, toy metric).\n"
    "- dependencies are python packages (e.g. numpy).\n"
    "- risks mirror validation.risks if any.\n"
    "Return JSON only."
)


def _fallback(artifact: dict) -> CodingSpec:
    demo_spec = (artifact or {}).get("demo_spec") or {}
    criteria = demo_spec.get("acceptance_criteria") or ["demo.py exits 0 on toy data"]
    validation = (artifact or {}).get("validation") or {}
    risks = validation.get("risks") or []
    # Build minimal file_tasks
    tasks = [
        FileTask(path="demo.py", goal="implement core demo", context_slice=demo_spec.get("core_algorithm", "")[:500] or demo_spec.get("pseudocode", "")[:500]),
        FileTask(path="requirements.txt", goal="dependencies", context_slice=",".join(demo_spec.get("tech_stack", ["python", "numpy"]))),
        FileTask(path="test_demo.py", goal="tests for acceptance criteria", context_slice="; ".join(criteria[:3])),
    ]
    test_intents = criteria[:3] if criteria else ["demo.py exits 0"]
    deps = demo_spec.get("tech_stack", ["python", "numpy"])
    # normalize deps to python packages (keep numpy etc, drop python)
    deps = [d for d in deps if d != "python"] or ["numpy"]
    return CodingSpec(file_tasks=tasks, test_intents=test_intents, dependencies=deps, risks=risks)


async def analyze_artifact(client, artifact) -> CodingSpec:
    # Normalize artifact to dict
    if hasattr(artifact, "model_dump"):
        artifact_dict = artifact.model_dump()
    elif isinstance(artifact, dict):
        artifact_dict = artifact
    else:
        artifact_dict = {}

    # Also handle RD ResearchArtifact model
    if hasattr(artifact, "demo_spec"):
        try:
            artifact_dict = artifact.model_dump() if hasattr(artifact, "model_dump") else dict(artifact)
        except Exception:
            pass

    payload = json.dumps(artifact_dict)[:6000]

    try:
        resp = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"ResearchArtifact:\n{payload}\n\nReturn CodingSpec JSON."},
            ],
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        # Validate and coerce
        tasks = []
        for t in data.get("file_tasks") or []:
            try:
                tasks.append(FileTask(path=str(t.get("path", "")), goal=str(t.get("goal", "")), context_slice=str(t.get("context_slice", ""))))
            except Exception:
                continue
        # Ensure fixed set
        paths = {t.path for t in tasks}
        for required in ["demo.py", "requirements.txt", "test_demo.py"]:
            if required not in paths:
                tasks.append(FileTask(path=required, goal=f"generate {required}", context_slice=""))

        test_intents = [str(x) for x in (data.get("test_intents") or [])][:5]
        if not test_intents:
            # fallback to acceptance_criteria
            demo_spec = artifact_dict.get("demo_spec") or {}
            criteria = demo_spec.get("acceptance_criteria") or []
            test_intents = [str(c) for c in criteria[:3]] or ["demo.py exits 0"]

        dependencies = [str(x) for x in (data.get("dependencies") or [])][:10]
        risks = [str(x) for x in (data.get("risks") or [])][:5]
        # If risks empty but validation has risks, use them
        if not risks:
            validation = artifact_dict.get("validation") or {}
            risks = [str(x) for x in (validation.get("risks") or [])][:5]

        return CodingSpec(file_tasks=tasks, test_intents=test_intents, dependencies=dependencies, risks=risks)
    except Exception:
        logger.exception("code analyzer failed, falling back")
        return _fallback(artifact_dict)
