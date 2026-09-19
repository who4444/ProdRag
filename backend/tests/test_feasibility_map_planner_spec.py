"""Codebase map (static + enrichment) and fixed 3-dimension planner (M2)."""

import pytest

from app.feasibility.codebase_map import enrich_map, static_map
from app.feasibility.planner import TAXONOMY, build_dimensions
from app.feasibility.schemas import DIMENSION_IDS, IdeaUnderstanding


def _fixture_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "eval").mkdir()
    (repo / "data").mkdir()
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "demo"\n[project.scripts]\ndemo = "pkg.cli:main"\n', encoding="utf-8"
    )
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "core.py").write_text(
        "from pkg.util import helper\n\n\nclass Engine:\n    pass\n\n\ndef run():\n    return helper()\n",
        encoding="utf-8",
    )
    (repo / "pkg" / "util.py").write_text("import os\n\n\ndef helper():\n    return os.getcwd()\n", encoding="utf-8")
    (repo / "pkg" / "cli.py").write_text(
        "from pkg.core import run\n\n\nif __name__ == '__main__':\n    run()\n", encoding="utf-8"
    )
    (repo / "docs.md").write_text("# docs\n", encoding="utf-8")
    (repo / "eval" / "golden.jsonl").write_text('{"q": "x"}\n', encoding="utf-8")
    (repo / "data" / "raw.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    return repo


def test_static_map_modules_edges_entrypoints_data(tmp_path):
    repo = _fixture_repo(tmp_path)
    cmap = static_map(repo, "add caching to the engine")

    paths = {m["path"] for m in cmap.modules}
    assert {"pkg/core.py", "pkg/util.py", "pkg/cli.py"} <= paths
    core = next(m for m in cmap.modules if m["path"] == "pkg/core.py")
    assert {"Engine", "run"} <= set(core["symbols"])

    assert ["pkg/core.py", "pkg/util.py"] in cmap.dependency_edges
    assert ["pkg/cli.py", "pkg/core.py"] in cmap.dependency_edges

    assert "pkg/cli.py" in cmap.entrypoints
    assert "[script] demo" in cmap.entrypoints
    assert "eval/golden.jsonl" in cmap.data_available
    assert "data/raw.csv" in cmap.data_available


def test_static_map_impacted_is_reverse_dependency_closure(tmp_path):
    repo = _fixture_repo(tmp_path)
    cmap = static_map(repo, "add caching to the engine")
    # Engine lives in pkg/core.py -> dependents (cli.py) are impacted too
    assert "pkg/core.py" in cmap.impacted_modules
    assert "pkg/cli.py" in cmap.impacted_modules


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

        async def _create(**kwargs):
            self.calls.append(kwargs)
            import json

            return type(
                "R",
                (),
                {"choices": [type("C", (), {"message": type("M", (), {"content": json.dumps(self.payload)})()})()]},
            )()

        self.chat = type("C", (), {"completions": type("CC", (), {"create": staticmethod(_create)})()})()


@pytest.mark.asyncio
async def test_enrich_map_validates_extension_paths_and_data(tmp_path):
    repo = _fixture_repo(tmp_path)
    cmap = static_map(repo, "add caching to the engine")
    client = _FakeClient(
        {
            "extension_points": [
                {"path": "pkg/core.py", "symbol": "Engine", "reason": "engine already owns state"},
                {"path": "does/not/exist.py", "symbol": "x", "reason": "bogus"},
            ],
            "data_flow": "csv -> engine -> output",
            "data_available": ["data/raw.csv"],
            "data_missing": ["labeled golden set for caching"],
            "data_cleanliness": "needs_cleaning",
        }
    )
    understanding = IdeaUnderstanding(interpretation="add a cache to the engine")

    out = await enrich_map(client, cmap, understanding, "add caching", repo, repo_id="r1")

    assert [ep.path for ep in out.extension_points] == ["pkg/core.py"]
    assert out.extension_points[0].evidence[0].valid is True
    assert out.data_flow == "csv -> engine -> output"
    assert "labeled golden set for caching" in out.data_missing
    assert out.data_cleanliness == "needs_cleaning"
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_enrich_map_falls_back_to_static_on_llm_failure(tmp_path):
    repo = _fixture_repo(tmp_path)
    cmap = static_map(repo, "add caching to the engine")
    baseline_data = list(cmap.data_available)

    class _Failing:
        def __init__(self):
            async def _create(**kwargs):
                raise RuntimeError("LLM down")

            self.chat = type("C", (), {"completions": type("CC", (), {"create": staticmethod(_create)})()})()

    out = await enrich_map(_Failing(), cmap, IdeaUnderstanding(), "add caching", repo)

    assert out.extension_points == []
    assert out.data_flow == ""
    assert out.data_available == baseline_data


def test_planner_fixed_three_dimensions():
    dims = build_dimensions(IdeaUnderstanding(interpretation="add caching"), "raw idea")
    assert [d["id"] for d in dims] == list(DIMENSION_IDS) == ["technical", "integration", "data"]
    assert len(TAXONOMY) == 3
    assert all("add caching" in d["question"] for d in dims)
