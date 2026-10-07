# mini-antemortem-cli - Easy Start

Short version of [README.md](README.md). Korean easy version: [EASY_README_KR.md](EASY_README_KR.md). Korean main README: [README_KR.md](README_KR.md).

## Start here · Standalone use · Integration/Docking

**Omega Aile** — Quiet precision. AI research guided by evidence.

Lint calibration config for deterministic structural traps without calling a model, then optionally pass its findings to the omegaprompt adaptation plan.

Requires Python 3.11+. Installation needs internet.

```bash
python -m pip install mini-antemortem-cli==0.10.1
mini-antemortem-cli --version
mini-antemortem-cli list-traps --json
```

The offline registry prints the built-in trap records and exits 0. For actual config analysis use check with train/test JSONL, rubric and variants JSON; the existing demo fixtures are linked below. Advisory check can exit 0 with serious findings; --fail-on-severity enables the policy gate (exit 1). Bad configuration exits 2.

omegaprompt>=1.1.0 is required and installed automatically; a calibration run is optional. The supported current combination is omegaprompt 2.1.2. Validate check JSON findings as AnalyticalFinding objects, wrap them in PreflightReport, then derive_adaptation_plan. This is separate from antemortem, which verifies repository citations.

[Docking contracts and runnable data handoff](https://github.com/hibou04-ops/omega-lock/blob/main/DOCKING.md) · [Full guide](README.md).

MCP: install the distribution with `[mcp]` and use its existing server executable. FastMCP support is bounded to MCP SDK `>=1.0.0,<2.0.0`; the tool names and schemas are unchanged.


## What Is This?

`mini-antemortem-cli` is a deterministic linter that catches the silent traps in your prompt-eval setup — train/test leakage, judge bias, homogeneous variants — before they fake a passing score. It reads your local `omegaprompt` calibration config inputs, checks 11 built-in trap patterns, and returns `AnalyticalFinding` records. No API keys, no live providers, no network. Works with `omegaprompt`; useful standalone as a config linter.

Repository: `hibou04-ops/mini-antemortem-cli` · PyPI: `mini-antemortem-cli` · import: `mini_antemortem_cli` · CLI: `mini-antemortem-cli` · MCP: `mini-antemortem-cli-mcp` with `mini-antemortem-cli[mcp]`

## What's New in 0.10.0

- Two new trap rules — the count is now 11 (was 9): `few_shot_leakage_into_test` (a few-shot example baked into the prompt matches a held-out item) and `rubric_dead_weight_dimension` (a rubric dimension with zero weight that the judge still scores).
- Claim-drift fix: the trap count is now consistent everywhere (it had said *nine* in one place and *seven* in another); the source of truth is `analytical_traps()` and a consistency check fails the build on drift.
- Version-agnostic publish workflow: the release tag, `pyproject` version, and `__init__` version must match before a build is published.

## What's New in 0.9.1

- Bad-input error messages are now a single line (schema errors used to span several lines). Exit code is still `2`.

## What's New in 0.9.0

- Text output now leads with one grep-friendly `Summary:` line (native status: PASS / ADVISORY / HOLD / BLOCK / NEEDS_MORE_EVIDENCE).
- High-signal findings carry the value the trap fired on in a `cite` field (config-referenced).
- Bad input files report a clean one-line error and exit code `2` (no traceback).
- `list-traps --json` emits a `{id, hypothesis}` array.
- Exact train/test ID overlap now fires at `BLOCKER` (was high). `--fail-on-severity high` still catches it.
- Moves to `4 - Beta`, with an additive-only commitment to the surface going forward.

## Install

```bash
pip install mini-antemortem-cli
```

For MCP:

```bash
pip install "mini-antemortem-cli[mcp]"
mini-antemortem-cli-mcp
```

## Trap IDs

- `self_agreement_bias`
- `small_sample_kc4_power`
- `variants_homogeneous`
- `rubric_weight_concentration`
- `judge_budget_too_small`
- `empty_reference_with_strict_rubric`
- `no_held_out_slice`
- `train_test_id_overlap`
- `routed_provider_opaque_family`
- `few_shot_leakage_into_test`
- `rubric_dead_weight_dimension`

The source-backed count and hypotheses are generated here: [docs/generated/claims.md](docs/generated/claims.md).

## Quick CLI Demo

```bash
python examples/demo_replay.py
```

Or run the CLI directly:

```bash
mini-antemortem-cli check \
  --target-provider openai \
  --target-model gpt-4o \
  --judge-provider openai \
  --judge-model gpt-4o \
  --train examples/demo_config/train.jsonl \
  --test examples/demo_config/test.jsonl \
  --rubric examples/demo_config/rubric.json \
  --variants examples/demo_config/variants.json \
  --json
```

## Python API

```python
from mini_antemortem_cli import analytical_preflight, analytical_traps
```

Use `analytical_preflight(...)` before calibration, then pass the findings into `omegaprompt.preflight.PreflightReport` and `derive_adaptation_plan`.

## When To Use It

- You want a deterministic sanity check before spending provider calls.
- You want CI to fail on `--fail-on-severity high`.
- You want explicit trap IDs you can diff over time.

## When To Use Something Else

- Use `mini-omega-lock` when you need empirical live/mock provider probes.
- Use `antemortem-cli` when you need broader source-file recon and disk-verified file:line citations.
- Use `omegaprompt` for the actual calibration engine.

License: Apache 2.0.
