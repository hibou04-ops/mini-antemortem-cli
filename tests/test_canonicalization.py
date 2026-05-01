"""Reviewer 4순위: trap classifier semantic strengthening.

Three classes of canonicalization were broken:

1. provider/model raw-string compare missed ``OpenAI`` vs ``openai``,
   ``azure-openai`` vs ``openai`` (same family, different label).
2. judge_output_budget == "small" only matched lowercase string;
   missed ``OutputBudgetBucket.SMALL`` (enum) and ``"SMALL"`` (case).
3. empty_reference fired NEW on every reference-less dataset, even
   when the rubric was self-contained.

Each test pins one of those.
"""

from __future__ import annotations

import pytest

from omegaprompt.domain.dataset import Dataset, DatasetItem
from omegaprompt.domain.enums import OutputBudgetBucket
from omegaprompt.domain.judge import Dimension, HardGate, JudgeRubric
from omegaprompt.domain.params import PromptVariants

from mini_antemortem_cli.traps import analytical_preflight


def _by_trap(findings, trap_id):
    return next(f for f in findings if f.trap_id == trap_id)


def _ds(n: int, with_ref: bool = True) -> Dataset:
    items = [
        DatasetItem(
            id=f"t{i}",
            input=f"in {i}",
            reference=f"ref {i}" if with_ref else None,
        )
        for i in range(n)
    ]
    return Dataset(items=items)


def _rubric_self_contained() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[
            Dimension(name="acc", description="is correct", weight=1.0),
        ],
        hard_gates=[HardGate(name="g0", description="g", evaluator="judge")],
    )


def _rubric_many_axes() -> JudgeRubric:
    """Rubric with > 5 axes total to trigger the budget-too-small trap."""
    dims = [
        Dimension(name=f"d{i}", description=f"dim {i}", weight=1.0)
        for i in range(4)
    ]
    gates = [
        HardGate(name=f"g{i}", description=f"gate {i}", evaluator="judge")
        for i in range(3)
    ]
    return JudgeRubric(dimensions=dims, hard_gates=gates)


def _variants() -> PromptVariants:
    return PromptVariants(
        system_prompts=[
            "You are precise.",
            "You are concise. Short answers only.",
            "You are careful. Double-check before replying.",
        ],
        few_shot_examples=[{"input": "1+1", "output": "2"}],
    )


# ---------------------------------------------------------------------------
# Provider canonicalization.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target,judge",
    [
        ("OpenAI", "openai"),       # case difference
        ("openai", "OpenAI"),
        ("azure-openai", "openai"), # family alias
        ("Azure-OpenAI", "openai"),
        ("anthropic", "claude"),    # claude is anthropic family
        ("vertex-ai", "google"),    # google family
        ("gemini", "vertex"),
    ],
)
def test_self_agreement_real_when_provider_family_matches(target, judge):
    findings = analytical_preflight(
        target_provider=target,
        target_model="m1",
        judge_provider=judge,
        judge_model="m2",  # different model -> not "identical", but same family
        train_dataset=_ds(20),
        test_dataset=_ds(15),
        rubric=_rubric_self_contained(),
        variants=_variants(),
    )
    f = _by_trap(findings, "self_agreement_bias")
    assert f.label == "REAL", f"expected REAL for {target}/{judge}, got {f.label}: {f.note}"


def test_self_agreement_ghost_for_different_families():
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt",
        judge_provider="anthropic",
        judge_model="claude",
        train_dataset=_ds(20),
        test_dataset=_ds(15),
        rubric=_rubric_self_contained(),
        variants=_variants(),
    )
    f = _by_trap(findings, "self_agreement_bias")
    assert f.label == "GHOST"


def test_self_agreement_high_when_provider_and_model_identical_after_canonical():
    findings = analytical_preflight(
        target_provider="OpenAI",
        target_model="GPT-4o",
        judge_provider="openai",
        judge_model="gpt-4o",
        train_dataset=_ds(20),
        test_dataset=_ds(15),
        rubric=_rubric_self_contained(),
        variants=_variants(),
    )
    f = _by_trap(findings, "self_agreement_bias")
    assert f.label == "REAL"
    assert f.severity.value == "high"


# ---------------------------------------------------------------------------
# Budget enum normalization.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "budget",
    [
        "small",
        "SMALL",
        "Small",
        OutputBudgetBucket.SMALL,
        "OutputBudgetBucket.SMALL",
    ],
)
def test_judge_budget_too_small_matches_all_budget_representations(budget):
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt",
        judge_provider="anthropic",
        judge_model="claude",
        train_dataset=_ds(20),
        test_dataset=_ds(15),
        rubric=_rubric_many_axes(),  # > 5 axes
        variants=_variants(),
        judge_output_budget=budget,
    )
    f = _by_trap(findings, "judge_budget_too_small")
    assert f.label == "REAL", f"expected REAL for budget={budget!r}, got {f.label}"


@pytest.mark.parametrize(
    "budget",
    [
        "medium",
        "MEDIUM",
        OutputBudgetBucket.MEDIUM,
        "large",
        OutputBudgetBucket.LARGE,
    ],
)
def test_judge_budget_ghost_when_budget_above_small(budget):
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt",
        judge_provider="anthropic",
        judge_model="claude",
        train_dataset=_ds(20),
        test_dataset=_ds(15),
        rubric=_rubric_many_axes(),
        variants=_variants(),
        judge_output_budget=budget,
    )
    f = _by_trap(findings, "judge_budget_too_small")
    assert f.label == "GHOST", f"expected GHOST for budget={budget!r}, got {f.label}"


# ---------------------------------------------------------------------------
# Rubric ground-truth keyword scan (multi-language).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "description",
    [
        "Does the answer match the expected output?",
        "Score against the ground truth.",
        "Compare to the reference answer.",
        "정답과 일치하는가?",
        "기준답안 대비 평가.",
    ],
)
def test_empty_reference_real_when_rubric_keyword_implies_ground_truth(description):
    rubric = JudgeRubric(
        dimensions=[
            Dimension(name="acc", description=description, weight=1.0),
        ],
        hard_gates=[HardGate(name="g0", description="g", evaluator="judge")],
    )
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt",
        judge_provider="anthropic",
        judge_model="claude",
        train_dataset=_ds(20, with_ref=False),
        test_dataset=_ds(15, with_ref=False),
        rubric=rubric,
        variants=_variants(),
    )
    f = _by_trap(findings, "empty_reference_with_strict_rubric")
    assert f.label == "REAL"
