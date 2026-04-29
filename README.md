# mini-antemortem-cli

> **Analytical preflight for [omegaprompt](https://pypi.org/project/omegaprompt/) calibration.** Reads run config, classifies seven calibration-specific trap patterns against deterministic rules. **No API calls, no network** — reasoning is deterministic given inputs. Emits `AnalyticalFinding` records that feed omegaprompt's `derive_adaptation_plan`.

[![PyPI](https://img.shields.io/badge/pypi-0.3.0-blue.svg)](https://pypi.org/project/mini-antemortem-cli/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![Parent](https://img.shields.io/badge/parent-omegaprompt%E2%89%A51.4.0-blueviolet.svg)](https://pypi.org/project/omegaprompt/)

> **Part of the omegaprompt toolkit** — [omegaprompt](https://github.com/hibou04-ops/omegaprompt) (calibration engine) · [omega-lock](https://github.com/hibou04-ops/omega-lock) (audit framework) · [antemortem-cli](https://github.com/hibou04-ops/antemortem-cli) (pre-implementation recon CLI) · [mini-omega-lock](https://github.com/hibou04-ops/mini-omega-lock) (empirical preflight) · [mini-antemortem-cli](https://github.com/hibou04-ops/mini-antemortem-cli) (analytical preflight, this repo) · [Antemortem](https://github.com/hibou04-ops/Antemortem) (methodology). Cross-toolkit cookbook: [AGENT_TRIGGERS.md](https://github.com/hibou04-ops/omegaprompt/blob/main/AGENT_TRIGGERS.md).

```bash
pip install omegaprompt mini-antemortem-cli
```

**MCP server.** This package also exposes its analytical classifier as agent-callable MCP tools (`analytical_preflight`, `list_traps`). Run `pip install "mini-antemortem-cli[mcp]"` then `python -m mini_antemortem_cli.mcp` (stdio, default for Claude Code). Deterministic, zero LLM cost. See [AGENT_TRIGGERS.md scenario 2](https://github.com/hibou04-ops/omegaprompt/blob/main/AGENT_TRIGGERS.md#scenario-2--pre-calibration-sanity-check).

---

## TL;DR

`omegaprompt` ships a **plugin interface** for preflight (`omegaprompt.preflight.contracts` + `omegaprompt.preflight.adaptation`) but no classifier code. This package fills that gap with seven deterministic trap classifiers — fully offline, fully reproducible:

- **`self_agreement_bias`** — target and judge share a vendor; judge biases overlap with target.
- **`small_sample_kc4_power`** — dataset too small for Pearson correlation to carry statistical power.
- **`variants_homogeneous`** — system-prompt variants too similar for sensitivity to have signal.
- **`rubric_weight_concentration`** — single rubric dimension carries most of the weight.
- **`judge_budget_too_small`** — judge output budget is SMALL but rubric has many dimensions + gates.
- **`empty_reference_with_strict_rubric`** — no dataset item has a reference; rubric implies ground-truth comparison.
- **`no_held_out_slice`** — no `--test` slice; walk-forward cannot run.

Each pattern returns `REAL` / `GHOST` / `NEW` / `UNRESOLVED` with a severity (`blocker` / `high` / `medium` / `low`) and a remediation hint.

> **Looking for the empirical (LLM-probe) preflight?** See sibling tool [`mini-omega-lock`](https://pypi.org/project/mini-omega-lock/) — same plugin interface, runs actual provider calls instead of static analysis.

---

## Quick start (1-minute, fully offline)

```python
from omegaprompt.domain.dataset import Dataset, DatasetItem
from omegaprompt.domain.judge import Dimension, HardGate, JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.preflight import PreflightReport, derive_adaptation_plan
from mini_antemortem_cli import analytical_preflight

# Your run config
rubric = JudgeRubric(
    dimensions=[Dimension(name="accuracy", description="x", weight=1.0)],
    hard_gates=[HardGate(name="no_violation", description="y")],
)
dataset = Dataset(items=[DatasetItem(id="ex1", input="2+2", reference="4")])
variants = PromptVariants(...)

# Run all seven trap classifiers — fully offline, fully deterministic
findings = analytical_preflight(dataset=dataset, rubric=rubric, variants=variants)

# Feed into omegaprompt's adaptation layer
report = PreflightReport(analytical_findings=findings)
plan = derive_adaptation_plan(report)
print(plan.recommendations)
```

No API keys, no network, no LLM calls — output is fully reproducible given inputs.

> 👋 Simpler intro: [EASY_README.md](EASY_README.md) (English) · [EASY_README_KR.md](EASY_README_KR.md)

---

## Why this is separate from omegaprompt

`omegaprompt` ships a **plugin interface** (`omegaprompt.preflight.contracts` + `omegaprompt.preflight.adaptation`) but no classifier code. Standalone users do not need analytical preflight — the main pipeline runs with declared defaults. Users who want analytical risk assessment over their configuration install this package alongside:

```bash
pip install omegaprompt mini-antemortem-cli
```

## Trap patterns

Seven deterministic classifications run against the run config:

| Trap id | Hypothesis |
|---|---|
| `self_agreement_bias` | Target and judge share a vendor; judge's biases overlap with target. |
| `small_sample_kc4_power` | Dataset too small for Pearson correlation to carry statistical power. |
| `variants_homogeneous` | System-prompt variants are too similar for sensitivity to have signal. |
| `rubric_weight_concentration` | A single rubric dimension carries most of the weight. |
| `judge_budget_too_small` | Judge output budget is SMALL but rubric has many dimensions + gates. |
| `empty_reference_with_strict_rubric` | No dataset item has a reference; rubric implies ground-truth comparison. |
| `no_held_out_slice` | No `--test` slice; walk-forward cannot run. |

Each pattern returns one of `REAL` / `GHOST` / `NEW` / `UNRESOLVED` with a severity (`blocker` / `high` / `medium` / `low`) and a remediation hint.

## Usage

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
    hard_gates=[HardGate(name="no_refusal", description="x", evaluator="judge")],
)
variants = PromptVariants(system_prompts=["You are an assistant."], few_shot_examples=[])
train = Dataset(items=[DatasetItem(id=f"t{i}", input=f"task {i}") for i in range(5)])
test = Dataset(items=[DatasetItem(id=f"v{i}", input=f"val {i}") for i in range(3)])

findings = analytical_preflight(
    target_provider="openai",
    target_model="gpt-4o-mini",
    judge_provider="openai",
    judge_model="gpt-4o-mini",
    train_dataset=train,
    test_dataset=test,
    rubric=rubric,
    variants=variants,
    judge_output_budget="small",
)

report = PreflightReport(analytical_findings=findings)
plan = derive_adaptation_plan(report=report)
# plan.skip_axes, plan.max_gap_override, etc.
```

## Design principles

- **Deterministic.** Same config in, same findings out. No LLM calls; no sampling noise.
- **Source-level citations.** Each finding carries a `note` describing the rule that fired and a `remediation` hint. No hand-wave.
- **Severity discipline.** `high` severity findings drive `AdaptationPlan` overrides that *only strengthen* the discipline (per `apply_adaptation_plan` invariants).
- **Extensible.** Add your own `TrapPattern` and classifier function; compose with the built-in seven.

## Validation

Every trap pattern has positive + negative test cases. No API calls, fully offline. Run with `pytest -q`.

## Relation to the family

- **[Antemortem](https://github.com/hibou04-ops/Antemortem)** / **[antemortem-cli](https://pypi.org/project/antemortem/)** — pre-implementation reconnaissance discipline for code changes. The naming "mini-antemortem-cli" echoes this family; the *enumerate-then-classify* pattern comes from there.
- **[omegaprompt](https://pypi.org/project/omegaprompt/)** — prompt calibration engine. This package feeds its preflight plugin interface.
- **[mini-omega-lock](https://pypi.org/project/mini-omega-lock/)** — empirical sibling. Runs live probes to measure judge consistency and endpoint reliability.

## License

Apache 2.0. See [LICENSE](LICENSE).

**License history.** PyPI distributions of version 0.1.0 were shipped with an MIT `LICENSE` file. The repository was relicensed to Apache 2.0 on 2026-04-22 (commit `d2d7eb7`); 0.2.0 (2026-04-28) and all later versions ship under Apache 2.0. Anyone who installed 0.1.0 holds an MIT license to that copy — license changes do not apply retroactively.
