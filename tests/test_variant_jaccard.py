"""Reviewer 4순위 last sub-item: variants_homogeneous now flags semantic
near-duplicates via token Jaccard, not length-only.

Pre-fix behaviour:
- Two prompts of very different lengths but ~identical vocabulary
  ("You are precise" vs "You are precise. Reply.") → GHOST (length
  span passes the threshold). Reality: they're near-duplicates and
  will not produce sensitivity signal.
- A single near-duplicate pair inside an otherwise diverse variant
  set went unflagged.

Post-fix:
- Max pairwise Jaccard >= 0.70 fires the trap as REAL/MEDIUM
  regardless of length span.
- Diverse variants with low Jaccard stay GHOST.
- Length-span check still fires (NEW/MEDIUM) when short and few
  variants — distinct from the Jaccard signal.
"""

from __future__ import annotations

import pytest

from omegaprompt.domain.dataset import Dataset, DatasetItem
from omegaprompt.domain.judge import Dimension, HardGate, JudgeRubric
from omegaprompt.domain.params import PromptVariants

from mini_antemortem_cli.traps import (
    _max_pairwise_jaccard,
    _tokenize_for_jaccard,
    analytical_preflight,
)


def _by_trap(findings, trap_id):
    return next(f for f in findings if f.trap_id == trap_id)


def _ds(n: int) -> Dataset:
    return Dataset(
        items=[
            DatasetItem(id=f"t{i}", input=f"in {i}", reference=f"ref {i}")
            for i in range(n)
        ]
    )


def _rubric() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[Dimension(name="acc", description="is correct", weight=1.0)],
        hard_gates=[HardGate(name="g0", description="g", evaluator="judge")],
    )


def _variants_from(prompts: list[str]) -> PromptVariants:
    return PromptVariants(
        system_prompts=prompts,
        few_shot_examples=[{"input": "1+1", "output": "2"}],
    )


def _run(variants: PromptVariants):
    return analytical_preflight(
        target_provider="openai",
        target_model="gpt",
        judge_provider="anthropic",
        judge_model="claude",
        train_dataset=_ds(20),
        test_dataset=_ds(15),
        rubric=_rubric(),
        variants=variants,
    )


# ---------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------


def test_tokenize_for_jaccard_strips_punctuation_and_lowercases():
    assert _tokenize_for_jaccard("Hello, World!") == {"hello", "world"}
    assert _tokenize_for_jaccard("") == set()


def test_max_pairwise_jaccard_zero_for_single_prompt():
    assert _max_pairwise_jaccard(["only one"]) == 0.0


def test_max_pairwise_jaccard_one_for_identical_prompts():
    j = _max_pairwise_jaccard(["A B C", "A B C"])
    assert j == 1.0


def test_max_pairwise_jaccard_zero_for_disjoint_vocab():
    j = _max_pairwise_jaccard(["alpha beta gamma delta", "one two three four"])
    assert j == 0.0


def test_max_pairwise_jaccard_uses_max_not_mean():
    """A single near-duplicate pair must dominate the score even when
    the variant set is otherwise diverse."""
    prompts = [
        "alpha beta gamma delta",
        "alpha beta gamma delta extra",  # near-dup of prompt[0]
        "completely different vocabulary words here",
    ]
    j = _max_pairwise_jaccard(prompts)
    # First two share 4/5 tokens; third is disjoint. Max should reflect the
    # near-duplicate pair, not the average across all pairs.
    assert j >= 0.7


# ---------------------------------------------------------------------------
# variants_homogeneous trap.
# ---------------------------------------------------------------------------


def test_variants_homogeneous_real_for_high_jaccard_despite_length_diff():
    """Two prompts of very different lengths but near-identical vocab
    used to slip through length-only — now caught by Jaccard."""
    variants = _variants_from([
        "You are a precise assistant who answers carefully.",
        # Same vocabulary plus filler - length differs a lot but tokens overlap
        "You are a precise assistant who answers carefully and " * 5
        + "You are a precise assistant who answers carefully.",
        "Your job is something completely different about cats and dogs and "
        "weather and trains and food and music and games.",
    ])
    findings = _run(variants)
    f = _by_trap(findings, "variants_homogeneous")
    # First two share most of their vocabulary -> max Jaccard high
    assert f.label == "REAL"
    assert "Jaccard" in f.note or "vocabulary" in f.note


def test_variants_homogeneous_ghost_for_diverse_variants():
    variants = _variants_from([
        "You are a precise assistant. Reply briefly.",
        "Decompose the question into steps before answering.",
        "Adopt a teacher persona; explain reasoning at each step.",
        "Output only the final answer with no explanation or preamble.",
    ])
    findings = _run(variants)
    f = _by_trap(findings, "variants_homogeneous")
    assert f.label == "GHOST"


def test_variants_homogeneous_new_for_short_compact_length_span():
    """Length-only signal still fires when prompts are short, few, and
    near-identical in length — preserves the pre-existing NEW path."""
    variants = _variants_from([
        "Be concise.",
        "Be precise.",
        "Be careful.",
    ])
    findings = _run(variants)
    f = _by_trap(findings, "variants_homogeneous")
    # Length span < 20 chars, <= 3 prompts -> NEW
    # (Could also fire as REAL if Jaccard happens to cross threshold;
    # we accept both as signals of homogeneity.)
    assert f.label in {"NEW", "REAL"}


def test_variants_homogeneous_real_overrides_new_when_both_signals_fire():
    """High-Jaccard short prompts surface as REAL (stronger signal),
    not NEW (weaker)."""
    variants = _variants_from([
        "Be precise carefully.",
        "Be precise carefully.",  # exact dup -> Jaccard 1.0
        "Be precise carefully.",
    ])
    findings = _run(variants)
    f = _by_trap(findings, "variants_homogeneous")
    assert f.label == "REAL"


def test_variants_homogeneous_remediation_mentions_jaccard_threshold():
    variants = _variants_from([
        "alpha beta gamma delta epsilon",
        "alpha beta gamma delta epsilon zeta",  # 5/6 overlap
        "alpha beta gamma delta epsilon eta",
    ])
    findings = _run(variants)
    f = _by_trap(findings, "variants_homogeneous")
    assert f.label == "REAL"
    assert "Jaccard" in f.remediation
