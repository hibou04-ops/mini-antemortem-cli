"""Analytical preflight (mini-antemortem adapter).

Reads the run configuration and emits classifications over a fixed set
of calibration-specific trap patterns. No API calls; all reasoning is
deterministic given the config.

The full antemortem-cli ``--domain calibration`` integration is a separate
project that uses an LLM to reason over the trap list against richer
evidence (vendor docs, prior-run artifacts, model cards). The in-process
version below covers the highest-signal traps with deterministic rules so
the core pipeline can stand alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from omegaprompt.domain.dataset import Dataset
from omegaprompt.domain.judge import JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.preflight.contracts import (
    AnalyticalFinding,
    PreflightSeverity,
)


# ----- Reviewer 4순위: canonicalization helpers -----


def _canonical_provider(name: str | None) -> str:
    """Lower-case, trim, normalize separators.

    Without this, ``OpenAI`` vs ``openai`` and ``azure_openai`` vs
    ``azure-openai`` would be different vendors to the self-agreement
    check, masking bias overlap.
    """
    if not name:
        return ""
    return str(name).strip().lower().replace("_", "-")


# Vendor families: providers that share underlying model weights or
# training data, so a self-agreement bias check should treat them as
# the same vendor for the purpose of "do they likely flatter each
# other's responses?". Conservative: only families where the public
# evidence is strong (azure-openai literally serves OpenAI models).
_PROVIDER_FAMILY: dict[str, str] = {
    "openai": "openai",
    "azure-openai": "openai",
    "azure": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "google": "google",
    "gemini": "google",
    "vertex": "google",
    "vertex-ai": "google",
}


def _provider_family(name: str | None) -> str:
    """Return the canonical family or the canonicalized name as-is."""
    canon = _canonical_provider(name)
    return _PROVIDER_FAMILY.get(canon, canon)


def _canonical_budget(budget: object) -> str:
    """Reduce any reasonable budget representation to ``small | medium | large``.

    Pre-fix: ``judge_output_budget == "small"`` only matched lowercase
    string literals. ``OutputBudgetBucket.SMALL`` (enum), ``"SMALL"``,
    ``"OutputBudgetBucket.SMALL"`` all silently passed the check
    because none of them equalled ``"small"``.
    """
    if budget is None:
        return ""
    raw = str(getattr(budget, "value", budget)).strip().lower()
    # Handle ``OutputBudgetBucket.SMALL`` -> ``small``.
    if "." in raw:
        raw = raw.rsplit(".", 1)[-1]
    return raw


# Reference-implying keywords (multi-language). When a rubric dimension
# description hits one of these, we treat the rubric as ground-truth-
# dependent — empty references then become a real trap rather than a
# benign self-contained rubric.
_REFERENCE_KEYWORDS: tuple[str, ...] = (
    "reference",
    "ground truth",
    "ground-truth",
    "expected output",
    "expected answer",
    "correct answer",
    "matches the",
    "compare to",
    "compared to",
    "정답",
    "기준답안",
    "참조",
)


def _rubric_implies_ground_truth(rubric: JudgeRubric) -> bool:
    """True when at least one dimension's description hints at a reference.

    Used to scope the ``empty_reference_with_strict_rubric`` trap to
    rubrics that actually depend on a reference; self-contained rubrics
    (e.g. \"is the response polite?\") get a GHOST instead of NEW.
    """
    for dim in rubric.dimensions:
        desc = (dim.description or "").lower()
        if any(kw in desc for kw in _REFERENCE_KEYWORDS):
            return True
    return False


@dataclass(frozen=True)
class TrapPattern:
    """One reusable calibration trap pattern."""

    id: str
    hypothesis: str


CALIBRATION_TRAPS: tuple[TrapPattern, ...] = (
    TrapPattern(
        id="self_agreement_bias",
        hypothesis=(
            "Target and judge share a vendor; judge's biases overlap with the "
            "target, flattering same-vendor responses."
        ),
    ),
    TrapPattern(
        id="small_sample_kc4_power",
        hypothesis=(
            "Dataset is small enough that Pearson KC-4 has no statistical power; "
            "correlation threshold becomes a random pass/fail."
        ),
    ),
    TrapPattern(
        id="variants_homogeneous",
        hypothesis=(
            "System prompt variants are too similar; sensitivity on the "
            "system_prompt_variant axis will be artificially low."
        ),
    ),
    TrapPattern(
        id="rubric_weight_concentration",
        hypothesis=(
            "A single rubric dimension carries the majority of the weight; "
            "judge noise on that one dimension dominates the fitness."
        ),
    ),
    TrapPattern(
        id="judge_budget_too_small",
        hypothesis=(
            "Judge output budget is SMALL but rubric has many dimensions + "
            "gates; judge response may be truncated before scoring all axes."
        ),
    ),
    TrapPattern(
        id="empty_reference_with_strict_rubric",
        hypothesis=(
            "Dataset items have no reference text while the rubric's "
            "dimension descriptions imply comparison to a ground truth."
        ),
    ),
    TrapPattern(
        id="no_held_out_slice",
        hypothesis=(
            "User did not pass --test; walk-forward cannot run; ship "
            "decision has no generalisation evidence."
        ),
    ),
)


def analytical_traps() -> tuple[TrapPattern, ...]:
    """Return the built-in calibration trap patterns."""
    return CALIBRATION_TRAPS


def _finding(
    trap: TrapPattern,
    *,
    label: str,
    severity: PreflightSeverity = PreflightSeverity.MEDIUM,
    note: str = "",
    remediation: str = "",
    cite: str | None = None,
) -> AnalyticalFinding:
    return AnalyticalFinding(
        trap_id=trap.id,
        label=label,
        hypothesis=trap.hypothesis,
        severity=severity,
        note=note,
        remediation=remediation,
        cite=cite,
    )


def _check_self_agreement(
    target_provider: str,
    target_model: str | None,
    judge_provider: str,
    judge_model: str | None,
) -> AnalyticalFinding:
    trap = next(t for t in CALIBRATION_TRAPS if t.id == "self_agreement_bias")

    # Canonicalize both names and pull them through the family map so
    # ``Azure-OpenAI`` and ``openai`` collapse to the same vendor for
    # bias purposes. Models compare canonically (lowercased + trimmed).
    target_family = _provider_family(target_provider)
    judge_family = _provider_family(judge_provider)
    target_model_norm = (target_model or "").strip().lower() or None
    judge_model_norm = (judge_model or "").strip().lower() or None

    if (
        target_family == judge_family
        and target_family != ""
        and target_model_norm == judge_model_norm
    ):
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.HIGH,
            note=(
                f"Target and judge are identical: {target_provider}/{target_model or 'default'}. "
                "Judge will share the target's distributional biases."
            ),
            remediation="Use a different vendor or stronger model for --judge-*.",
        )
    if target_family == judge_family and target_family != "":
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.MEDIUM,
            note=(
                f"Target and judge share vendor family ({target_family}); "
                "some bias overlap. Inputs were "
                f"target={target_provider!r}, judge={judge_provider!r}."
            ),
            remediation="Consider a cross-vendor judge to break self-agreement bias.",
        )
    return _finding(
        trap,
        label="GHOST",
        severity=PreflightSeverity.LOW,
        note=f"Target on {target_provider}, judge on {judge_provider} - different vendors.",
    )


def _check_sample_power(train_size: int, test_size: int | None) -> AnalyticalFinding:
    trap = next(t for t in CALIBRATION_TRAPS if t.id == "small_sample_kc4_power")
    total = train_size + (test_size or 0)
    if test_size is None:
        # KC-4 won't run; separate trap handles that.
        return _finding(
            trap,
            label="GHOST",
            severity=PreflightSeverity.LOW,
            note="No --test slice provided; Pearson check will not execute.",
        )
    if test_size < 10:
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.HIGH,
            note=(
                f"Test slice has {test_size} items. Pearson correlation at n={test_size} "
                "has weak statistical power; KC-4 pass/fail may be random."
            ),
            remediation=(
                "Expand test set to at least 20 items, or raise --min-kc4 adaptively "
                "(handled by AdaptationPlan)."
            ),
        )
    if total < 20:
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.MEDIUM,
            note=f"Total dataset is {total} items; noise absorption limited.",
            remediation="Larger datasets yield more reliable calibration gradients.",
        )
    return _finding(
        trap,
        label="GHOST",
        severity=PreflightSeverity.LOW,
        note=f"Dataset size {total} adequate for Pearson power.",
    )


def _tokenize_for_jaccard(text: str) -> set[str]:
    """Lowercased word-level token set used for Jaccard similarity.

    Whitespace split with light punctuation stripping. Deliberately simple:
    we only need a Pareto signal that two prompts share most of their
    vocabulary, not a real NLP tokenizer.
    """
    if not text:
        return set()
    cleaned = text.lower()
    for ch in ".,;:!?\"'`()[]{}<>":
        cleaned = cleaned.replace(ch, " ")
    return {tok for tok in cleaned.split() if tok}


def _max_pairwise_jaccard(prompts: list[str]) -> float:
    """Highest pairwise Jaccard similarity over the prompt list.

    Returns 0.0 for fewer than 2 prompts. The max (rather than mean) is
    used so a single near-duplicate pair flags the trap even when other
    variants are diverse.
    """
    if len(prompts) < 2:
        return 0.0
    token_sets = [_tokenize_for_jaccard(p) for p in prompts]
    best = 0.0
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            a, b = token_sets[i], token_sets[j]
            if not a and not b:
                continue
            union = a | b
            if not union:
                continue
            jaccard = len(a & b) / len(union)
            if jaccard > best:
                best = jaccard
    return best


# Threshold above which variants are considered semantic near-duplicates.
# 0.7 is permissive — variants legitimately share role/system framing
# tokens. Above this they're mostly the same prompt with cosmetic edits.
_JACCARD_NEAR_DUPLICATE = 0.70


def _check_variants_homogeneity(variants: PromptVariants) -> AnalyticalFinding:
    trap = next(t for t in CALIBRATION_TRAPS if t.id == "variants_homogeneous")
    prompts = variants.system_prompts
    if len(prompts) < 2:
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.MEDIUM,
            note="Only one system-prompt variant; axis contributes zero search signal.",
            remediation="Provide at least 3 genuinely distinct system prompts.",
        )
    lengths = [len(p) for p in prompts]
    length_span_small = max(lengths) - min(lengths) < 20 and len(prompts) <= 3
    max_jaccard = _max_pairwise_jaccard(list(prompts))
    high_token_overlap = max_jaccard >= _JACCARD_NEAR_DUPLICATE

    # Trap fires when EITHER signal triggers — length compactness OR
    # token overlap. Reviewer 4순위 last sub-item: length alone misses
    # cases where two prompts of very different lengths share most of
    # their vocabulary (e.g. one prompt is the other plus a closing
    # sentence), and length-different but semantically-identical
    # variants don't produce meaningful sensitivity.
    if high_token_overlap:
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.MEDIUM,
            note=(
                f"System-prompt variants share {max_jaccard:.0%} of vocabulary "
                f"(token Jaccard) at their most-similar pair; even with "
                f"{min(lengths)}-{max(lengths)} char span, variants are likely "
                "near-duplicates and will not produce meaningful sensitivity."
            ),
            remediation=(
                "Author variants that differ in role framing, instruction style, "
                "or task decomposition — not just wording. Aim for max pairwise "
                f"Jaccard below {_JACCARD_NEAR_DUPLICATE:.0%}."
            ),
        )
    if length_span_small:
        return _finding(
            trap,
            label="NEW",
            severity=PreflightSeverity.MEDIUM,
            note=(
                f"All {len(prompts)} system prompts have near-identical length "
                f"({min(lengths)}-{max(lengths)} chars, max Jaccard "
                f"{max_jaccard:.0%}); they may be too similar to produce "
                "meaningful sensitivity."
            ),
            remediation=(
                "Author variants that differ in role framing, not just wording; "
                "vary length by at least 2x where possible."
            ),
        )
    return _finding(
        trap,
        label="GHOST",
        severity=PreflightSeverity.LOW,
        note=(
            f"System-prompt variants span {min(lengths)}-{max(lengths)} chars "
            f"(max pairwise Jaccard {max_jaccard:.0%}); sufficient diversity "
            "expected."
        ),
    )


def _check_rubric_concentration(rubric: JudgeRubric) -> AnalyticalFinding:
    trap = next(t for t in CALIBRATION_TRAPS if t.id == "rubric_weight_concentration")
    weights = rubric.normalized_weights()
    if not weights:
        return _finding(trap, label="UNRESOLVED")
    max_w = max(weights.values())
    max_name = max(weights, key=weights.get)  # type: ignore[arg-type]
    if max_w > 0.7:
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.MEDIUM,
            note=(
                f"Dimension '{max_name}' carries {max_w:.0%} of the rubric weight; "
                "judge noise on that single dimension will dominate fitness."
            ),
            remediation=(
                "Rebalance rubric so no single dimension exceeds ~50% weight, "
                "or explicitly declare this concentration is intentional."
            ),
        )
    return _finding(
        trap,
        label="GHOST",
        severity=PreflightSeverity.LOW,
        note=f"Max dimension weight is {max_w:.0%}; no single-dim dominance.",
    )


def _check_judge_budget(rubric: JudgeRubric, judge_output_budget: object) -> AnalyticalFinding:
    trap = next(t for t in CALIBRATION_TRAPS if t.id == "judge_budget_too_small")
    n_dims = len(rubric.dimensions)
    n_gates = len([g for g in rubric.hard_gates if g.evaluator == "judge"])
    total_axes = n_dims + n_gates
    # Accept ``"small"``, ``"SMALL"``, ``OutputBudgetBucket.SMALL``,
    # and ``"OutputBudgetBucket.SMALL"`` — anything that resolves to
    # ``small`` after canonicalization. Pre-fix this only matched the
    # exact lowercase string literal and silently let enum / mixed-case
    # callers slip through.
    canon = _canonical_budget(judge_output_budget)
    if canon == "small" and total_axes > 5:
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.MEDIUM,
            note=(
                f"Judge budget SMALL (1024 tokens) vs {total_axes} rubric axes "
                f"({n_dims} dims + {n_gates} gates). JudgeResult response may be truncated."
            ),
            remediation="Raise LLMJudge output_budget to MEDIUM or reduce rubric surface.",
        )
    return _finding(
        trap,
        label="GHOST",
        severity=PreflightSeverity.LOW,
        note=f"Judge budget {canon or judge_output_budget!r} adequate for {total_axes} axes.",
    )


def _check_empty_reference(dataset: Dataset, rubric: JudgeRubric) -> AnalyticalFinding:
    trap = next(t for t in CALIBRATION_TRAPS if t.id == "empty_reference_with_strict_rubric")
    has_ref = sum(1 for it in dataset.items if it.reference)
    total = len(dataset.items)
    rubric_needs_reference = _rubric_implies_ground_truth(rubric)
    # Trap only fires when the rubric's descriptions actually imply a
    # ground-truth comparison AND the dataset has no references. A
    # self-contained rubric (\"is the response polite?\") with no
    # references is fine — pre-fix that case got flagged as NEW.
    if has_ref == 0 and total > 0 and rubric_needs_reference:
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.MEDIUM,
            note=(
                "No dataset item has a reference field, yet the rubric's dimension "
                "descriptions imply comparison to a ground truth. Judge will score "
                "without the anchor the rubric expects."
            ),
            remediation=(
                "Add reference fields, or reword the rubric so each dimension is "
                "self-contained (no \"compare to expected output\" language)."
            ),
        )
    if has_ref == 0 and total > 0:
        return _finding(
            trap,
            label="GHOST",
            severity=PreflightSeverity.LOW,
            note=(
                "No references in dataset, but rubric dimensions appear "
                "self-contained — no ground-truth anchor required."
            ),
        )
    return _finding(
        trap,
        label="GHOST",
        severity=PreflightSeverity.LOW,
        note=f"{has_ref}/{total} items carry reference text.",
    )


def _check_no_held_out(has_test_slice: bool) -> AnalyticalFinding:
    trap = next(t for t in CALIBRATION_TRAPS if t.id == "no_held_out_slice")
    if not has_test_slice:
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.HIGH,
            note=(
                "No --test slice was provided. Walk-forward validation will not run "
                "and the artifact's ship recommendation will be HOLD regardless of "
                "training fitness."
            ),
            remediation="Provide --test <held_out.jsonl> for walk-forward validation.",
        )
    return _finding(
        trap,
        label="GHOST",
        severity=PreflightSeverity.LOW,
        note="Held-out slice provided; walk-forward will run.",
    )


def analytical_preflight(
    *,
    target_provider: str,
    target_model: str | None,
    judge_provider: str,
    judge_model: str | None,
    train_dataset: Dataset,
    test_dataset: Dataset | None,
    rubric: JudgeRubric,
    variants: PromptVariants,
    judge_output_budget: str = "small",
) -> list[AnalyticalFinding]:
    """Run analytical preflight checks and return one finding per trap.

    All checks are deterministic given the inputs. The ordering of the
    returned list is stable (same order as :data:`CALIBRATION_TRAPS`).
    """
    findings: list[AnalyticalFinding] = [
        _check_self_agreement(target_provider, target_model, judge_provider, judge_model),
        _check_sample_power(
            train_size=len(train_dataset),
            test_size=len(test_dataset) if test_dataset is not None else None,
        ),
        _check_variants_homogeneity(variants),
        _check_rubric_concentration(rubric),
        _check_judge_budget(rubric, judge_output_budget),
        _check_empty_reference(train_dataset, rubric),
        _check_no_held_out(has_test_slice=test_dataset is not None),
    ]
    return findings
