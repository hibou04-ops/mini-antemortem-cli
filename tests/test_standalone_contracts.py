"""The standalone classifier retains input validation without its engine."""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from mini_antemortem_cli._vendor import (
    Dataset,
    Dimension,
    JudgeRubric,
    PromptVariants,
)
from mini_antemortem_cli.bridge import BridgeUnavailableError, to_omegaprompt_findings


ROOT = Path(__file__).resolve().parents[1]


def test_engine_is_optional_in_distribution_metadata():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert not any(dep.startswith("omegaprompt") for dep in project["dependencies"])
    assert project["optional-dependencies"]["bridge"] == ["omegaprompt>=1.1.0"]


@pytest.mark.parametrize("args", [["--help"], ["list-traps", "--json"]])
def test_cli_runs_when_engine_import_is_forbidden(args):
    code = """
import importlib.abc
import runpy
import sys
class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'omegaprompt' or fullname.startswith('omegaprompt.'):
            raise AssertionError('standalone CLI imported its optional engine')
sys.meta_path.insert(0, NoEngine())
sys.argv = ['mini-antemortem-cli', *sys.argv[1:]]
runpy.run_module('mini_antemortem_cli.cli', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, *args], capture_output=True, text=True,
        encoding="utf-8", check=False,
    )
    assert result.returncode == 0, result.stderr
    if args[0] == "list-traps":
        assert len(json.loads(result.stdout)) == 11


@pytest.mark.parametrize("args", [["--help"], ["check", "--help"]])
def test_installed_console_help_works_on_cp949_without_utf8_mode(args):
    executable = Path(sys.executable).parent / (
        "mini-antemortem-cli.exe" if os.name == "nt" else "mini-antemortem-cli"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONUTF8", None)
    environment["PYTHONIOENCODING"] = "cp949"
    result = subprocess.run(
        [str(executable), *args], capture_output=True, text=True,
        encoding="cp949", env=environment, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_dataset_duplicate_ids_and_empty_inputs_keep_validation():
    with pytest.raises(ValueError, match="duplicate ids"):
        Dataset.from_items([{"id": "a", "input": "one"}, {"id": "a", "input": "two"}])
    with pytest.raises(ValueError, match="zero items"):
        Dataset.from_items([])
    with pytest.raises(ValidationError):
        Dataset.from_items([{"id": "a", "input": ""}])


def test_rubric_and_variants_keep_original_validation():
    dimension = Dimension(name="quality", description="correctness", weight=1)
    with pytest.raises(ValidationError, match="unique names"):
        JudgeRubric(dimensions=[dimension, dimension])
    with pytest.raises(ValidationError):
        Dimension(name="quality", description="correctness", weight=1, scale=(5, 1))
    with pytest.raises(ValidationError, match="non-empty"):
        PromptVariants(system_prompts=["valid"], few_shot_examples=[{"input": "x"}])


def test_missing_optional_bridge_reports_install_command(monkeypatch):
    original_import = builtins.__import__

    def without_engine(name, *args, **kwargs):
        if name == "omegaprompt" or name.startswith("omegaprompt."):
            raise ModuleNotFoundError("No module named 'omegaprompt'", name="omegaprompt")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_engine)
    with pytest.raises(BridgeUnavailableError, match=r"mini-antemortem-cli\[bridge\]"):
        to_omegaprompt_findings([])


def test_mcp_accepts_compatible_foreign_model_classes():
    from mini_antemortem_cli.mcp.server import (
        _resolve_dataset,
        _resolve_rubric,
        _resolve_variants,
    )

    class ForeignDataset(BaseModel):
        items: list[dict]

    class ForeignRubric(BaseModel):
        dimensions: list[dict]
        hard_gates: list[dict] = []

    class ForeignVariants(BaseModel):
        system_prompts: list[str]
        few_shot_examples: list[dict] = []

    assert _resolve_dataset(ForeignDataset(items=[{"id": "a", "input": "x"}])).items[0].id == "a"
    rubric = _resolve_rubric(ForeignRubric(dimensions=[
        {"name": "quality", "description": "correctness", "weight": 1},
    ]))
    assert rubric.normalized_weights() == {"quality": 1.0}
    assert _resolve_variants(ForeignVariants(system_prompts=["prompt"])).system_prompts == ["prompt"]
