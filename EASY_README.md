# mini-antemortem-cli — Easy Start

> The short version, for people who found the main README intimidating.
> Full doc: [README.md](README.md) · 한국어 Easy: [EASY_README_KR.md](EASY_README_KR.md)

## What is this?

An optional plug-in for [omegaprompt](https://pypi.org/project/omegaprompt/) that **sanity-checks your calibration config before you run it** — against 7 known calibration-specific trap patterns. Pure deterministic rules. No API calls. No network. Runs in ~1 ms.

Its sibling [mini-omega-lock](https://pypi.org/project/mini-omega-lock/) measures your *environment* with live API calls. This one just reads your *config* and tells you where it looks structurally unsafe.

## Install

```bash
pip install mini-antemortem-cli
```

Requires `omegaprompt>=1.1.0` (imports the preflight contract schema).

## The 7 trap classifiers

Each one is a pure function against your config. Returns `REAL` (rule fired), `GHOST` (config is fine on this axis), `NEW` (edge case surfaced), or `UNRESOLVED`.

| # | Trap id | The concrete rule | Severity when it fires |
|---|---|---|---|
| 1 | `self_agreement_bias` | target provider **and** model == judge provider and model | HIGH (both match) / MEDIUM (same provider, different model) |
| 2 | `small_sample_kc4_power` | test size < 10, or train+test < 20 | HIGH (< 10) / MEDIUM (< 20 total) |
| 3 | `variants_homogeneous` | < 2 prompts, or ≤3 prompts with length range < 20 chars | MEDIUM |
| 4 | `rubric_weight_concentration` | single dimension weight > 70% | MEDIUM |
| 5 | `judge_budget_too_small` | budget = SMALL **and** (dimensions + gates) > 5 | MEDIUM |
| 6 | `empty_reference_with_strict_rubric` | no item has a `reference` field | LOW |
| 7 | `no_held_out_slice` | no test dataset provided | HIGH |

Every finding includes a concrete `note` (which rule fired) and a `remediation` (what to do). `BLOCKER` severity aborts guarded runs; `HIGH` triggers automatic AdaptationPlan overrides (e.g., skip the sensitivity axis for homogeneous variants).

## The minimum working example

```python
from omegaprompt.domain.dataset import Dataset, DatasetItem
from omegaprompt.domain.judge import Dimension, HardGate, JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.preflight import PreflightReport, derive_adaptation_plan
from mini_antemortem_cli import analytical_preflight

rubric = JudgeRubric(
    dimensions=[
        Dimension(name="accuracy", description="correct", weight=0.85),
        Dimension(name="clarity",  description="readable", weight=0.15),
    ],
    hard_gates=[HardGate(name="no_refusal", description="...", evaluator="judge")],
)
variants = PromptVariants(system_prompts=["You are an assistant."], few_shot_examples=[])
train = Dataset(items=[DatasetItem(id=f"t{i}", input=f"task {i}") for i in range(5)])
test  = Dataset(items=[DatasetItem(id=f"v{i}", input=f"val {i}") for i in range(3)])

findings = analytical_preflight(
    target_provider="openai", target_model="gpt-4o-mini",
    judge_provider="openai",  judge_model="gpt-4o-mini",   # <-- same vendor + same model = trap #1 fires
    train_dataset=train,
    test_dataset=test,
    rubric=rubric,                                          # <-- weight 0.85 on one dim = trap #4 fires
    variants=variants,                                      # <-- single prompt = trap #3 fires
    judge_output_budget="small",
)

for f in findings:
    print(f"{f.trap_id}: {f.label} ({f.severity}) — {f.note}")

# Feed into omegaprompt's adaptation layer:
report = PreflightReport(analytical_findings=findings)
plan   = derive_adaptation_plan(report=report)
# plan.skip_axes, plan.max_gap_override, etc.
```

**No network.** **No API keys.** **Same config → same findings, always.**

## The 4 exports

```python
from mini_antemortem_cli import (
    analytical_preflight,   # the composite — runs all 7 classifiers
    analytical_traps,       # returns tuple of the 7 TrapPattern definitions
    CALIBRATION_TRAPS,      # the tuple itself (for inspection)
    TrapPattern,            # dataclass: id + hypothesis
)
```

## When to use it

- You're about to run a real calibration and want a 1 ms gut-check first.
- You want CI to block configs with structural traps (e.g., accidentally using the same model for target and judge).
- You want deterministic findings you can diff across config changes.

## When to skip it

- You already know your config is sane (same project, unchanged for weeks).
- You're in rapid iteration and don't want even 1 ms of friction.

## Composes with mini-omega-lock

Both plug into the same `PreflightReport`:

```python
from mini_omega_lock import empirical_preflight
from mini_antemortem_cli import analytical_preflight

jq, ep, perf = empirical_preflight(...)           # live API measurements
findings      = analytical_preflight(...)          # deterministic rules

report = PreflightReport(
    judge_quality=jq, endpoint=ep, performance=perf,
    analytical_findings=findings,
)
plan = derive_adaptation_plan(report=report)
```

Empirical measures what your environment *actually does*. Analytical catches what your config *structurally risks*. Complementary, not redundant.

## Go deeper

- Classifier implementations: `src/mini_antemortem_cli/traps.py`
- Finding contract: `omegaprompt.preflight.contracts.AnalyticalFinding`
- Severity → AdaptationPlan mapping: `omegaprompt.preflight.adaptation.derive_adaptation_plan`
- Sibling empirical preflight (with live API probes): [mini-omega-lock](https://pypi.org/project/mini-omega-lock/)

License: Apache 2.0. Copyright (c) 2026 hibou.
