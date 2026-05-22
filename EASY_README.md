# mini-antemortem-cli - Easy Start

Short version of [README.md](README.md). Korean easy version: [EASY_README_KR.md](EASY_README_KR.md).

## What Is This?

`mini-antemortem-cli` is a deterministic preflight checker for `omegaprompt` calibration configs. It reads your local config inputs, checks 9 built-in trap patterns, and returns `AnalyticalFinding` records. No API keys, no live providers, no network.

Repository: `hibou04-ops/mini-antemortem-cli` · PyPI: `mini-antemortem-cli` · import: `mini_antemortem_cli` · CLI: `mini-antemortem-cli` · MCP: `mini-antemortem-cli-mcp` with `mini-antemortem-cli[mcp]`

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
