"""Reviewer C: real MCP execution tests for mini-antemortem-cli.

The pre-existing test_mcp_server.py covered tool registration shape and
``list_traps`` execution but never invoked ``analytical_preflight``
through the MCP layer with real on-disk inputs. This file calls the
registered MCP tool with temp-file JSONL/JSON inputs and asserts the
executed return value contract — particularly that all 7 trap findings
flow through the boundary intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omegaprompt.domain.dataset import Dataset, DatasetItem
from omegaprompt.domain.judge import Dimension, HardGate, JudgeRubric
from omegaprompt.domain.params import PromptVariants


@pytest.fixture(autouse=True)
def _workspace_root_to_tmp(monkeypatch, tmp_path):
    """All MCP execution tests use ``tmp_path`` for their dataset / rubric
    files. The workspace boundary added in audit item #8 defaults to cwd,
    so we pin ``MINI_ANTEMORTEM_WORKSPACE_ROOT`` to ``tmp_path`` here for
    every test in this file."""
    monkeypatch.setenv("MINI_ANTEMORTEM_WORKSPACE_ROOT", str(tmp_path))


@pytest.fixture
def mcp_server():
    from mini_antemortem_cli.mcp import server as srv

    return srv


def _write_dataset(path: Path, n: int, with_ref: bool = True) -> None:
    items = [
        DatasetItem(
            id=f"t{i}",
            input=f"task {i}",
            reference=f"ref {i}" if with_ref else None,
        )
        for i in range(n)
    ]
    path.write_text(
        "\n".join(it.model_dump_json() for it in items),
        encoding="utf-8",
    )


def _rubric_dict() -> dict:
    return {
        "dimensions": [
            {"name": "acc", "description": "is correct", "weight": 0.7},
            {"name": "clarity", "description": "is clear", "weight": 0.3},
        ],
        "hard_gates": [
            {"name": "no_refusal", "description": "must try", "evaluator": "judge"},
        ],
    }


def _variants_dict() -> dict:
    return {
        "system_prompts": [
            "You are a precise assistant.",
            "You are a concise assistant. Reply briefly and accurately.",
            "You are a careful assistant who double-checks before replying.",
        ],
        "few_shot_examples": [{"input": "1+1", "output": "2"}],
    }


# ---------------------------------------------------------------------------
# analytical_preflight execution.
# ---------------------------------------------------------------------------


def test_mcp_analytical_preflight_returns_nine_findings(mcp_server, tmp_path):
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    _write_dataset(train, n=15, with_ref=True)
    _write_dataset(test, n=15, with_ref=True)

    findings = mcp_server.analytical_preflight(
        target_provider="openai",
        judge_provider="anthropic",
        train_dataset=str(train),
        rubric=_rubric_dict(),
        variants=_variants_dict(),
        target_model="gpt-4o",
        judge_model="claude-opus-4-7",
        test_dataset=str(test),
        judge_output_budget="medium",
    )
    # JSON-encodable through the MCP boundary:
    json.dumps(findings)
    # All 9 traps must appear (8th: train_test_id_overlap from earlier
    # batch; 9th: routed_provider_opaque_family from item #6 — fires
    # GHOST/LOW for non-routed providers, UNRESOLVED/MEDIUM otherwise).
    expected = {
        "self_agreement_bias",
        "small_sample_kc4_power",
        "variants_homogeneous",
        "rubric_weight_concentration",
        "judge_budget_too_small",
        "empty_reference_with_strict_rubric",
        "no_held_out_slice",
        "train_test_id_overlap",
        "routed_provider_opaque_family",
    }
    actual = {f["trap_id"] for f in findings}
    assert actual == expected


def test_mcp_analytical_preflight_flags_no_held_out_when_test_absent(mcp_server, tmp_path):
    train = tmp_path / "train.jsonl"
    _write_dataset(train, n=15)

    findings = mcp_server.analytical_preflight(
        target_provider="openai",
        judge_provider="anthropic",
        train_dataset=str(train),
        rubric=_rubric_dict(),
        variants=_variants_dict(),
    )
    no_held_out = next(f for f in findings if f["trap_id"] == "no_held_out_slice")
    assert no_held_out["label"] == "REAL"
    assert no_held_out["severity"] == "high"


def test_mcp_analytical_preflight_self_agreement_real_for_same_family(mcp_server, tmp_path):
    """Reviewer 4순위 verification: provider canonicalization works
    end-to-end through MCP — Azure-OpenAI and openai are the same family."""
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    _write_dataset(train, n=15)
    _write_dataset(test, n=15)

    findings = mcp_server.analytical_preflight(
        target_provider="Azure-OpenAI",
        judge_provider="openai",
        train_dataset=str(train),
        rubric=_rubric_dict(),
        variants=_variants_dict(),
        target_model="m1",
        judge_model="m2",
        test_dataset=str(test),
    )
    bias = next(f for f in findings if f["trap_id"] == "self_agreement_bias")
    assert bias["label"] == "REAL"


def test_mcp_analytical_preflight_handles_inline_rubric_dict(mcp_server, tmp_path):
    """Inline rubric/variants must work — agents typically pass dicts,
    not paths, when calling MCP tools from JSON-RPC."""
    train = tmp_path / "train.jsonl"
    _write_dataset(train, n=15)

    findings = mcp_server.analytical_preflight(
        target_provider="openai",
        judge_provider="anthropic",
        train_dataset=str(train),
        rubric=_rubric_dict(),  # dict, not path
        variants=_variants_dict(),  # dict, not path
    )
    assert len(findings) == 9


# ---------------------------------------------------------------------------
# list_traps execution.
# ---------------------------------------------------------------------------


def test_mcp_list_traps_returns_all_nine(mcp_server):
    traps = mcp_server.list_traps()
    json.dumps(traps)
    ids = {t["id"] for t in traps}
    assert ids == {
        "self_agreement_bias",
        "small_sample_kc4_power",
        "variants_homogeneous",
        "rubric_weight_concentration",
        "judge_budget_too_small",
        "empty_reference_with_strict_rubric",
        "no_held_out_slice",
        "train_test_id_overlap",
        "routed_provider_opaque_family",
    }
    for t in traps:
        assert "hypothesis" in t
        assert len(t["hypothesis"]) > 20  # real one-sentence description
