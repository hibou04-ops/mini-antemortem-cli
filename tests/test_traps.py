"""Analytical preflight tests - trap classifications over a range of configs."""

from __future__ import annotations

from omegaprompt.domain.dataset import Dataset, DatasetItem
from omegaprompt.domain.judge import Dimension, HardGate, JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.preflight.contracts import PreflightSeverity
from mini_antemortem_cli.traps import TrapPolicy, analytical_preflight, analytical_traps


def _rubric(dim_weights=None, gates=1) -> JudgeRubric:
    if dim_weights is None:
        dim_weights = {"accuracy": 0.7, "clarity": 0.3}
    return JudgeRubric(
        dimensions=[
            Dimension(name=name, description=f"{name} description", weight=w)
            for name, w in dim_weights.items()
        ],
        hard_gates=[
            HardGate(name=f"g{i}", description=f"gate {i}", evaluator="judge")
            for i in range(gates)
        ],
    )


def _variants(n_prompts=3, lens=(100, 200, 500)) -> PromptVariants:
    prompts = ["X" * lens[i % len(lens)] for i in range(n_prompts)]
    return PromptVariants(
        system_prompts=prompts,
        few_shot_examples=[{"input": "1+1", "output": "2"}],
    )


def _dataset(n=5, with_ref=False) -> Dataset:
    items = [
        DatasetItem(
            id=f"t{i}",
            input=f"task {i}",
            reference=f"ref {i}" if with_ref else None,
        )
        for i in range(n)
    ]
    return Dataset(items=items)


def _by_trap(findings, trap_id):
    return next(f for f in findings if f.trap_id == trap_id)


def test_trap_registry_contains_all_expected():
    trap_ids = {t.id for t in analytical_traps()}
    assert trap_ids == {
        "self_agreement_bias",
        "small_sample_kc4_power",
        "variants_homogeneous",
        "rubric_weight_concentration",
        "judge_budget_too_small",
        "empty_reference_with_strict_rubric",
        "no_held_out_slice",
        "train_test_id_overlap",
        "routed_provider_opaque_family",
        "few_shot_leakage_into_test",
        "rubric_dead_weight_dimension",
    }


def test_self_agreement_identical_is_real_high():
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt-4o",
        judge_provider="openai",
        judge_model="gpt-4o",
        train_dataset=_dataset(n=20, with_ref=True),
        test_dataset=_dataset(n=15, with_ref=True),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "self_agreement_bias")
    assert f.label == "REAL"
    assert f.severity == PreflightSeverity.HIGH


def test_self_agreement_same_vendor_different_model_is_real_medium():
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt-4o",
        judge_provider="openai",
        judge_model="gpt-4o-mini",
        train_dataset=_dataset(n=20, with_ref=True),
        test_dataset=_dataset(n=15, with_ref=True),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "self_agreement_bias")
    assert f.label == "REAL"
    assert f.severity == PreflightSeverity.MEDIUM


def test_self_agreement_cross_vendor_is_ghost():
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt-4o",
        judge_provider="anthropic",
        judge_model="claude-opus-4-7",
        train_dataset=_dataset(n=20, with_ref=True),
        test_dataset=_dataset(n=15, with_ref=True),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "self_agreement_bias")
    assert f.label == "GHOST"


def test_small_sample_power_flags_small_test_slice():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=5),
        test_dataset=_dataset(n=5),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "small_sample_kc4_power")
    assert f.label == "REAL"
    assert f.severity == PreflightSeverity.HIGH


def test_small_sample_power_ghost_when_no_test_slice():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=5),
        test_dataset=None,
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "small_sample_kc4_power")
    assert f.label == "GHOST"


def test_variants_homogeneous_flags_uniform_length_variants():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(),
        variants=PromptVariants(
            system_prompts=["You are an assistant.", "You are a helper."],
            few_shot_examples=[],
        ),
    )
    f = _by_trap(findings, "variants_homogeneous")
    assert f.label == "NEW"


def test_variants_single_prompt_is_real():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(),
        variants=PromptVariants(system_prompts=["only one"], few_shot_examples=[]),
    )
    f = _by_trap(findings, "variants_homogeneous")
    assert f.label == "REAL"


def test_rubric_weight_concentration_flags_over_70():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(dim_weights={"accuracy": 0.9, "clarity": 0.1}),
        variants=_variants(),
    )
    f = _by_trap(findings, "rubric_weight_concentration")
    assert f.label == "REAL"


def test_rubric_weight_concentration_ghost_on_balanced():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(dim_weights={"accuracy": 0.5, "clarity": 0.5}),
        variants=_variants(),
    )
    f = _by_trap(findings, "rubric_weight_concentration")
    assert f.label == "GHOST"


def test_judge_budget_too_small_flags_many_axes_on_small_budget():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(
            dim_weights={f"d{i}": 1.0 / 6 for i in range(6)},
            gates=2,
        ),
        variants=_variants(),
        judge_output_budget="small",
    )
    f = _by_trap(findings, "judge_budget_too_small")
    assert f.label == "REAL"


def test_judge_budget_ghost_on_medium_budget():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(),
        variants=_variants(),
        judge_output_budget="medium",
    )
    f = _by_trap(findings, "judge_budget_too_small")
    assert f.label == "GHOST"


def test_empty_reference_real_when_no_refs_and_rubric_implies_ground_truth():
    """Reviewer 4순위: trap fires only when the rubric actually requires
    a reference. Self-contained rubrics with no refs are GHOST."""
    rubric_needs_ref = JudgeRubric(
        dimensions=[
            Dimension(
                name="accuracy",
                description="Does the answer match the expected output?",
                weight=1.0,
            ),
        ],
        hard_gates=[HardGate(name="g0", description="g", evaluator="judge")],
    )
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20, with_ref=False),
        test_dataset=_dataset(n=15, with_ref=False),
        rubric=rubric_needs_ref,
        variants=_variants(),
    )
    f = _by_trap(findings, "empty_reference_with_strict_rubric")
    assert f.label == "REAL"


def test_empty_reference_ghost_when_rubric_self_contained():
    """Self-contained rubric ('is the response polite?') with no refs
    is fine — pre-fix this got flagged NEW."""
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20, with_ref=False),
        test_dataset=_dataset(n=15, with_ref=False),
        rubric=_rubric(),  # default _rubric: descriptions like "accuracy description"
        variants=_variants(),
    )
    f = _by_trap(findings, "empty_reference_with_strict_rubric")
    assert f.label == "GHOST"


def test_empty_reference_ghost_when_refs_present():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20, with_ref=True),
        test_dataset=_dataset(n=15, with_ref=True),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "empty_reference_with_strict_rubric")
    assert f.label == "GHOST"


def test_no_held_out_slice_real_when_missing():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=None,
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "no_held_out_slice")
    assert f.label == "REAL"
    assert f.severity == PreflightSeverity.HIGH


def test_no_held_out_slice_ghost_when_provided():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "no_held_out_slice")
    assert f.label == "GHOST"


# ---------------------------------------------------------------------------
# Reviewer P1 #6: routed-aggregator providers obscure family.
# OpenRouter / Together / Fireworks / Groq / Bedrock / etc. forward to a
# backend the family-collapse logic can't see. Surface as UNRESOLVED.
# ---------------------------------------------------------------------------


def test_routed_provider_unresolved_when_target_uses_openrouter():
    findings = analytical_preflight(
        target_provider="openrouter",
        target_model="meta-llama/llama-3-70b",
        judge_provider="anthropic",
        judge_model="claude",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "routed_provider_opaque_family")
    assert f.label == "UNRESOLVED"
    assert f.severity == PreflightSeverity.MEDIUM
    assert "openrouter" in f.note.lower()


def test_routed_provider_unresolved_when_judge_uses_bedrock():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="bedrock",
        judge_model="anthropic.claude-3",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "routed_provider_opaque_family")
    assert f.label == "UNRESOLVED"
    assert "bedrock" in f.note.lower()


def test_routed_provider_ghost_when_neither_is_routed():
    """First-party providers (anthropic / openai / google) on both sides
    leave this trap as GHOST — the family-collapse logic upstream has
    full visibility, so this trap doesn't add new signal."""
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "routed_provider_opaque_family")
    assert f.label == "GHOST"
    assert f.severity == PreflightSeverity.LOW


def test_routed_provider_canonicalization_handles_underscores():
    """together_ai → together-ai → matches frozenset entry."""
    findings = analytical_preflight(
        target_provider="Together_AI",
        target_model="meta",
        judge_provider="anthropic",
        judge_model="claude",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "routed_provider_opaque_family")
    assert f.label == "UNRESOLVED"


# ---------------------------------------------------------------------------
# C2 (0.9.0): config-referenced citations on high-signal traps. Each enriched
# trap's `cite` is non-None and substrings the firing value. Other traps leave
# cite None (existing AnalyticalFinding optional field; no schema change).
# ---------------------------------------------------------------------------


def _overlap_datasets():
    train = Dataset(
        items=[DatasetItem(id=i, input="x") for i in ["shared", "ta", "tb", "tc"]]
    )
    test = Dataset(
        items=[DatasetItem(id=i, input="x") for i in ["shared", "va", "vb", "vc"]]
    )
    return train, test


def test_cite_train_test_id_overlap_names_overlapping_id():
    train, test = _overlap_datasets()
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt-4o",
        judge_provider="anthropic",
        judge_model="claude-opus-4-7",
        train_dataset=train,
        test_dataset=test,
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "train_test_id_overlap")
    assert f.cite is not None
    assert "shared" in f.cite


def test_cite_rubric_weight_concentration_names_dimension_and_weight():
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt-4o",
        judge_provider="anthropic",
        judge_model="claude-opus-4-7",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric({"accuracy": 0.9, "clarity": 0.1}),
        variants=_variants(),
    )
    f = _by_trap(findings, "rubric_weight_concentration")
    assert f.cite is not None
    # Short form: "dimension: accuracy (90% of weight)".
    assert "accuracy" in f.cite
    assert "dimension" in f.cite


def test_cite_variants_homogeneous_reports_max_jaccard():
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt-4o",
        judge_provider="anthropic",
        judge_model="claude-opus-4-7",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(),
        variants=PromptVariants(
            system_prompts=[
                "You are a careful assistant that double checks every answer.",
                "You are a careful assistant that double checks every reply.",
            ],
            few_shot_examples=[{"input": "1+1", "output": "2"}],
        ),
    )
    f = _by_trap(findings, "variants_homogeneous")
    assert f.label == "REAL"  # high token overlap branch
    assert f.cite is not None
    assert "Jaccard" in f.cite


def test_cite_small_sample_power_reports_test_size():
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt-4o",
        judge_provider="anthropic",
        judge_model="claude-opus-4-7",
        train_dataset=_dataset(n=12),
        test_dataset=_dataset(n=4),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "small_sample_kc4_power")
    assert f.label == "REAL"
    assert f.cite is not None
    assert "4" in f.cite  # the firing test size


def test_non_enriched_traps_leave_cite_none():
    # A clean cross-vendor config: self_agreement is GHOST and carries no cite.
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt-4o",
        judge_provider="anthropic",
        judge_model="claude-opus-4-7",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(),
        variants=_variants(),
    )
    assert _by_trap(findings, "self_agreement_bias").cite is None


# ---------------------------------------------------------------------------
# H2 (0.9.0): exact train/test ID overlap is BLOCKER; summarize_findings -> BLOCK.
# ---------------------------------------------------------------------------


def test_overlap_emits_blocker_severity():
    train, test = _overlap_datasets()
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt-4o",
        judge_provider="anthropic",
        judge_model="claude-opus-4-7",
        train_dataset=train,
        test_dataset=test,
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "train_test_id_overlap")
    assert f.label == "REAL"
    assert f.severity == PreflightSeverity.BLOCKER


def test_overlap_summarizes_to_block_status():
    from mini_antemortem_cli.traps import summarize_findings

    train, test = _overlap_datasets()
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt-4o",
        judge_provider="anthropic",
        judge_model="claude-opus-4-7",
        train_dataset=train,
        test_dataset=test,
        rubric=_rubric(),
        variants=_variants(),
    )
    summary = summarize_findings(findings)
    assert summary["status"] == "BLOCK"
    assert summary["highest_severity"] == "blocker"


def test_within_slice_duplicates_stay_medium():
    # Within-slice dup (not cross-slice overlap) must remain MEDIUM, not BLOCKER.
    train = Dataset(items=[DatasetItem(id=i, input="x") for i in ["d", "d", "e"]])
    test = Dataset(items=[DatasetItem(id=i, input="x") for i in ["x", "y", "z"]])
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt-4o",
        judge_provider="anthropic",
        judge_model="claude-opus-4-7",
        train_dataset=train,
        test_dataset=test,
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "train_test_id_overlap")
    assert f.severity == PreflightSeverity.MEDIUM


# ---------------------------------------------------------------------------
# 0.10.0: few_shot_leakage_into_test
# ---------------------------------------------------------------------------


def _variants_with_shot(shot_input: str, shot_output: str = "answer") -> PromptVariants:
    return PromptVariants(
        system_prompts=["Answer.", "Explain first.", "Check edges then answer."],
        few_shot_examples=[{"input": shot_input, "output": shot_output}],
    )


def _kwargs(**over):
    base = dict(
        target_provider="openai",
        target_model="gpt-4o",
        judge_provider="anthropic",
        judge_model="claude-opus-4-7",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=20),
        rubric=_rubric(),
        variants=_variants(),
    )
    base.update(over)
    return base


def test_few_shot_leakage_into_test_is_real_high():
    test = Dataset(items=[DatasetItem(id="leak", input="What is 2+2?")] + [
        DatasetItem(id=f"v{i}", input=f"eval {i}") for i in range(19)
    ])
    findings = analytical_preflight(
        **_kwargs(test_dataset=test, variants=_variants_with_shot("What is 2+2?"))
    )
    f = _by_trap(findings, "few_shot_leakage_into_test")
    assert f.label == "REAL"
    assert f.severity == PreflightSeverity.HIGH
    assert "leak" in (f.cite or "")


def test_few_shot_leakage_into_train_only_is_real_medium():
    train = Dataset(items=[DatasetItem(id="trleak", input="What is 2+2?")] + [
        DatasetItem(id=f"t{i}", input=f"task {i}") for i in range(19)
    ])
    test = Dataset(items=[DatasetItem(id=f"v{i}", input=f"eval {i}") for i in range(20)])
    findings = analytical_preflight(
        **_kwargs(train_dataset=train, test_dataset=test, variants=_variants_with_shot("What is 2+2?"))
    )
    f = _by_trap(findings, "few_shot_leakage_into_test")
    assert f.label == "REAL"
    assert f.severity == PreflightSeverity.MEDIUM


def test_few_shot_leakage_matches_on_reference_text():
    # Overlap on the item's reference (not just input) also leaks the answer.
    test = Dataset(items=[DatasetItem(id="rleak", input="distinct", reference="42")] + [
        DatasetItem(id=f"v{i}", input=f"eval {i}") for i in range(19)
    ])
    findings = analytical_preflight(
        **_kwargs(test_dataset=test, variants=_variants_with_shot("prompt", shot_output="42"))
    )
    f = _by_trap(findings, "few_shot_leakage_into_test")
    assert f.label == "REAL"


def test_few_shot_leakage_ghost_when_disjoint():
    findings = analytical_preflight(**_kwargs(variants=_variants_with_shot("totally unrelated text")))
    f = _by_trap(findings, "few_shot_leakage_into_test")
    assert f.label == "GHOST"


def test_few_shot_leakage_ghost_when_no_examples():
    variants = PromptVariants(system_prompts=["a", "b", "c"], few_shot_examples=[])
    findings = analytical_preflight(**_kwargs(variants=variants))
    f = _by_trap(findings, "few_shot_leakage_into_test")
    assert f.label == "GHOST"


def test_few_shot_leakage_disabled_by_policy():
    test = Dataset(items=[DatasetItem(id="leak", input="What is 2+2?")] + [
        DatasetItem(id=f"v{i}", input=f"eval {i}") for i in range(19)
    ])
    findings = analytical_preflight(
        **_kwargs(test_dataset=test, variants=_variants_with_shot("What is 2+2?")),
        policy=TrapPolicy(check_few_shot_leakage=False),
    )
    f = _by_trap(findings, "few_shot_leakage_into_test")
    assert f.label == "GHOST"


# ---------------------------------------------------------------------------
# 0.10.0: rubric_dead_weight_dimension
# ---------------------------------------------------------------------------


def test_rubric_dead_weight_is_real_medium():
    findings = analytical_preflight(
        **_kwargs(rubric=_rubric(dim_weights={"accuracy": 1.0, "clarity": 0.0}))
    )
    f = _by_trap(findings, "rubric_dead_weight_dimension")
    assert f.label == "REAL"
    assert f.severity == PreflightSeverity.MEDIUM
    assert "clarity" in (f.cite or "")


def test_rubric_dead_weight_ghost_when_all_weighted():
    findings = analytical_preflight(
        **_kwargs(rubric=_rubric(dim_weights={"accuracy": 0.6, "clarity": 0.4}))
    )
    f = _by_trap(findings, "rubric_dead_weight_dimension")
    assert f.label == "GHOST"


def test_rubric_dead_weight_ghost_for_single_dimension():
    # A single weighted dimension is not "dead weight" — it carries everything.
    findings = analytical_preflight(**_kwargs(rubric=_rubric(dim_weights={"accuracy": 1.0})))
    f = _by_trap(findings, "rubric_dead_weight_dimension")
    assert f.label == "GHOST"
