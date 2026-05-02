# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""FastMCP server wrapping mini-antemortem-cli's analytical preflight classifier.

Deterministic, zero-LLM-cost. Catches seven well-known calibration traps
before a full omegaprompt run is paid for.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mini_antemortem_cli import (
    CALIBRATION_TRAPS,
    analytical_preflight as _analytical_preflight,
)


# ---------------------------------------------------------------------------
# Workspace boundary
# ---------------------------------------------------------------------------
#
# Reviewer audit item #8: MCP tools accept dataset / rubric / variants paths
# from agent input. Without a boundary, an adversarial / careless prompt
# could request ``"rubric": "/etc/passwd"`` and the loader would dutifully
# read it, leaking partial contents through the JSON-decode error message
# in the MCP response.
#
# Pattern matches mini-omega-lock's MINI_OMEGA_WORKSPACE_ROOT. When the env
# var is unset we default to cwd so the standard "agent invokes from project
# root" case works without configuration.


def _workspace_root() -> Path:
    """The directory all path inputs must resolve inside.

    Reads ``MINI_ANTEMORTEM_WORKSPACE_ROOT`` lazily so tests can monkeypatch
    per-call. Resolves to an absolute path so the comparison below is
    stable across symlinks and relative inputs.
    """
    raw = os.environ.get("MINI_ANTEMORTEM_WORKSPACE_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.cwd().resolve()


def _validate_workspace_path(candidate: Path) -> Path:
    """Reject paths that escape the configured workspace root.

    Raises ``ValueError`` (the MCP layer will propagate it as a
    structured error) when ``candidate`` resolves outside
    ``_workspace_root()``. Returns the resolved path on success so the
    caller can use it directly.
    """
    root = _workspace_root()
    resolved = candidate.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"path {str(candidate)!r} resolves to {str(resolved)!r}, "
            f"which is outside the configured workspace root "
            f"{str(root)!r}. Set MINI_ANTEMORTEM_WORKSPACE_ROOT to widen "
            f"the boundary, or pass an inline dict / list for the input "
            f"instead of a path."
        ) from exc
    return resolved

mcp_app = FastMCP(
    name="mini-antemortem-cli",
    instructions=(
        "Analytical preflight classifier for omegaprompt calibration. "
        "Deterministic rule-based classification of seven calibration trap "
        "patterns: self-agreement bias, small-sample KC-4 power, variant "
        "homogeneity, rubric concentration, judge budget, empty reference, "
        "missing held-out slice. Use BEFORE paying for a full calibrate() "
        "to surface cheap-to-fix configuration issues."
    ),
)


def _resolve_dataset(d):
    """Accept Dataset, path, or inline list[dict].

    Reviewer item #8: agents invoking MCP via JSON-RPC frequently want to
    pass items inline rather than write a JSONL file first. Inline
    list[dict] support means the agent can run a preflight on a synthetic
    probe dataset without round-tripping through the filesystem.
    """
    from omegaprompt.domain import Dataset, DatasetItem

    if isinstance(d, Dataset):
        return d
    if isinstance(d, list):
        # Inline list of items — each item is a DatasetItem-shaped dict.
        # No filesystem access; no workspace check needed.
        items = [
            entry if isinstance(entry, DatasetItem) else DatasetItem.model_validate(entry)
            for entry in d
        ]
        return Dataset(items=items)
    if isinstance(d, (str, Path)):
        # Path branch: bound to workspace root before reading.
        safe_path = _validate_workspace_path(Path(d))
        return Dataset.from_jsonl(safe_path)
    raise TypeError(f"Unsupported dataset input: {type(d).__name__}")


def _resolve_rubric(r):
    from omegaprompt.domain import JudgeRubric

    if isinstance(r, JudgeRubric):
        return r
    if isinstance(r, dict):
        # Inline dict — no filesystem access.
        return JudgeRubric.model_validate(r)
    if isinstance(r, (str, Path)):
        safe_path = _validate_workspace_path(Path(r))
        return JudgeRubric.from_json(safe_path)
    raise TypeError(f"Unsupported rubric input: {type(r).__name__}")


def _resolve_variants(v):
    from omegaprompt.domain import PromptVariants

    if isinstance(v, PromptVariants):
        return v
    if isinstance(v, dict):
        # Inline dict — no filesystem access.
        return PromptVariants.model_validate(v)
    if isinstance(v, (str, Path)):
        safe_path = _validate_workspace_path(Path(v))
        return PromptVariants.model_validate_json(
            safe_path.read_text(encoding="utf-8")
        )
    raise TypeError(f"Unsupported variants input: {type(v).__name__}")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp_app.tool()
def analytical_preflight(
    target_provider: str,
    judge_provider: str,
    train_dataset: str,
    rubric: str | dict,
    variants: str | dict,
    target_model: str | None = None,
    judge_model: str | None = None,
    test_dataset: str | None = None,
    judge_output_budget: str = "small",
) -> list[dict]:
    """Classify the calibration config against the seven trap patterns.

    Returns one AnalyticalFinding per matched trap (label REAL / GHOST /
    NEW / UNRESOLVED, severity, hypothesis, remediation hint, citation).
    Deterministic — zero LLM calls.

    Args:
        target_provider: Provider name for the target model.
        judge_provider: Provider name for the judge model.
        train_dataset: Path to train dataset JSONL.
        rubric: Path or inline dict for the JudgeRubric.
        variants: Path or inline dict for the PromptVariants.
        target_model: Specific target model (or None for provider default).
        judge_model: Specific judge model (or None for provider default).
        test_dataset: Path to held-out test dataset JSONL (recommended).
        judge_output_budget: ``small`` / ``medium`` / ``large``. Used by
            the judge-budget trap to flag insufficient response space.

    Returns:
        List of AnalyticalFinding dicts. Each has ``trap_id``, ``label``,
        ``hypothesis``, ``severity``, ``note``, ``remediation``, ``cite``.
    """
    rubric_obj = _resolve_rubric(rubric)
    variants_obj = _resolve_variants(variants)
    train = _resolve_dataset(train_dataset)
    test = _resolve_dataset(test_dataset) if test_dataset is not None else None

    findings = _analytical_preflight(
        target_provider=target_provider,
        target_model=target_model,
        judge_provider=judge_provider,
        judge_model=judge_model,
        train_dataset=train,
        test_dataset=test,
        rubric=rubric_obj,
        variants=variants_obj,
        judge_output_budget=judge_output_budget,
    )
    return [f.model_dump(mode="json") for f in findings]


@mcp_app.tool()
def list_traps() -> list[dict]:
    """Return the seven calibration trap patterns this classifier checks.

    Use to introspect what `analytical_preflight` will check, or to display
    a configuration-review checklist to a human reviewer.

    Returns:
        List of dicts, each with ``id`` (e.g. ``self_agreement_bias``) and
        ``hypothesis`` (one-sentence description of the trap).
    """
    return [{"id": t.id, "hypothesis": t.hypothesis} for t in CALIBRATION_TRAPS]
