# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Reviewer audit item #8: MCP path inputs respect MINI_ANTEMORTEM_WORKSPACE_ROOT,
and inline list/dict inputs skip path validation.

Pre-fix every path the agent supplied was passed straight to the loader.
``rubric="/etc/passwd"`` would surface the read failure or partial-parse
error through the MCP response. Now path inputs resolve against the
workspace root, and out-of-tree paths raise structured ValueError before
any disk read.

Inline list (datasets) and inline dict (rubric/variants) inputs are
unaffected — they don't touch the filesystem.
"""

from __future__ import annotations

import json
import pytest

mcp = pytest.importorskip("mcp")


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    """Pin MINI_ANTEMORTEM_WORKSPACE_ROOT to a tmpdir so each test owns
    its boundary."""
    monkeypatch.setenv("MINI_ANTEMORTEM_WORKSPACE_ROOT", str(tmp_path))
    return tmp_path


def _write_rubric(path):
    rubric = {
        "dimensions": [
            {"name": "q", "description": "q", "weight": 1.0, "scale": [0, 1]},
        ],
        "hard_gates": [],
    }
    path.write_text(json.dumps(rubric), encoding="utf-8")


def _write_variants(path):
    variants = {"system_prompts": ["p"], "few_shot_examples": []}
    path.write_text(json.dumps(variants), encoding="utf-8")


def _write_dataset(path, n=5):
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"id": f"t{i}", "input": f"task {i}"}) + "\n")


# ---------------------------------------------------------------------------
# Workspace boundary on path inputs
# ---------------------------------------------------------------------------


def test_rubric_path_inside_workspace_resolves_normally(workspace):
    from mini_antemortem_cli.mcp.server import _resolve_rubric

    rubric_path = workspace / "rubric.json"
    _write_rubric(rubric_path)
    result = _resolve_rubric(str(rubric_path))
    assert result.dimensions[0].name == "q"


def test_rubric_path_outside_workspace_raises(workspace, tmp_path_factory):
    from mini_antemortem_cli.mcp.server import _resolve_rubric

    outside = tmp_path_factory.mktemp("outside")
    outside_path = outside / "rubric.json"
    _write_rubric(outside_path)

    with pytest.raises(ValueError, match="outside the configured workspace root"):
        _resolve_rubric(str(outside_path))


def test_dataset_path_outside_workspace_raises(workspace, tmp_path_factory):
    from mini_antemortem_cli.mcp.server import _resolve_dataset

    outside = tmp_path_factory.mktemp("outside")
    outside_path = outside / "data.jsonl"
    _write_dataset(outside_path, n=3)

    with pytest.raises(ValueError, match="outside the configured workspace root"):
        _resolve_dataset(str(outside_path))


def test_variants_path_outside_workspace_raises(workspace, tmp_path_factory):
    from mini_antemortem_cli.mcp.server import _resolve_variants

    outside = tmp_path_factory.mktemp("outside")
    outside_path = outside / "variants.json"
    _write_variants(outside_path)

    with pytest.raises(ValueError, match="outside the configured workspace root"):
        _resolve_variants(str(outside_path))


def test_path_with_traversal_raises(workspace):
    """``..`` traversal that escapes the workspace root must raise."""
    from mini_antemortem_cli.mcp.server import _resolve_rubric

    traversal = workspace / ".." / ".." / "etc" / "passwd"
    with pytest.raises(ValueError, match="outside the configured workspace root"):
        _resolve_rubric(str(traversal))


def test_workspace_default_is_cwd_when_env_unset(monkeypatch, tmp_path):
    from mini_antemortem_cli.mcp.server import _resolve_rubric

    monkeypatch.delenv("MINI_ANTEMORTEM_WORKSPACE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    rubric_inside = tmp_path / "rubric.json"
    _write_rubric(rubric_inside)
    assert _resolve_rubric(str(rubric_inside)).dimensions[0].name == "q"


# ---------------------------------------------------------------------------
# Inline object inputs skip path validation
# ---------------------------------------------------------------------------


def test_inline_rubric_dict_bypasses_workspace_check(workspace):
    from mini_antemortem_cli.mcp.server import _resolve_rubric

    inline = {
        "dimensions": [
            {"name": "q", "description": "q", "weight": 1.0, "scale": [0, 1]},
        ],
        "hard_gates": [],
    }
    result = _resolve_rubric(inline)
    assert result.dimensions[0].name == "q"


def test_inline_variants_dict_bypasses_workspace_check(workspace):
    from mini_antemortem_cli.mcp.server import _resolve_variants

    inline = {"system_prompts": ["p"], "few_shot_examples": []}
    result = _resolve_variants(inline)
    assert result.system_prompts == ["p"]


def test_inline_dataset_list_bypasses_workspace_check(workspace):
    """Reviewer item #8: agents typically want to send probe items
    inline rather than write JSONL to disk first."""
    from mini_antemortem_cli.mcp.server import _resolve_dataset

    items = [
        {"id": "t1", "input": "first task"},
        {"id": "t2", "input": "second task"},
    ]
    result = _resolve_dataset(items)
    assert len(result.items) == 2
    assert result.items[0].id == "t1"


def test_inline_dataset_list_works_for_analytical_preflight(workspace):
    """End-to-end: inline list[dict] reaches `analytical_preflight` and
    produces the expected number of findings."""
    from mini_antemortem_cli.mcp import server as mcp_server

    items = [{"id": f"t{i}", "input": f"task {i}"} for i in range(15)]
    rubric = {
        "dimensions": [
            {"name": "q", "description": "q", "weight": 1.0, "scale": [0, 1]},
        ],
        "hard_gates": [],
    }
    variants = {"system_prompts": ["sp"], "few_shot_examples": []}
    findings = mcp_server.analytical_preflight(
        target_provider="openai",
        judge_provider="anthropic",
        train_dataset=items,  # inline list, not a path
        rubric=rubric,
        variants=variants,
    )
    assert len(findings) == 9
    assert all("trap_id" in f for f in findings)
