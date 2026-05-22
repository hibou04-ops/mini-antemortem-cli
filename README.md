# mini-antemortem-cli

Deterministic analytical preflight for `omegaprompt` calibration configs: it reads config inputs, classifies 9 source-backed built-in trap patterns, and emits `AnalyticalFinding` records without provider calls or network access.

[![CI](https://github.com/hibou04-ops/mini-antemortem-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/hibou04-ops/mini-antemortem-cli/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mini-antemortem-cli.svg?cb=2)](https://pypi.org/project/mini-antemortem-cli/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)

```bash
pip install mini-antemortem-cli
```

Repository: `hibou04-ops/mini-antemortem-cli` · PyPI: `mini-antemortem-cli` · import: `mini_antemortem_cli` · CLI: `mini-antemortem-cli` · MCP: `mini-antemortem-cli-mcp` with `mini-antemortem-cli[mcp]`

## Trust / Verification Links

- Generated source-of-truth claims: [English](docs/generated/claims.md) / [Korean](docs/generated/claims_kr.md)
- Trust model: [English](docs/trust_model.md) / [Korean](docs/trust_model_kr.md)
- Toolkit positioning: [English](docs/toolkit_positioning.md) / [Korean](docs/toolkit_positioning_kr.md)
- Claim ledger: [English](docs/claim_ledger.md) / [Korean](docs/claim_ledger_kr.md)
- Examples and deterministic demo: [English](docs/examples.md) / [Korean](docs/examples_kr.md)
- Simpler README: [English](EASY_README.md) / [Korean](EASY_README_KR.md)
- CLI exit codes: [docs/cli_exit_codes.md](docs/cli_exit_codes.md)
- Release checklist: [docs/release_checklist.md](docs/release_checklist.md)
- Post-release verification: [docs/post_release_verification.md](docs/post_release_verification.md)

## Use It When

- You are about to run an `omegaprompt` calibration and want a deterministic structural check first.
- You want CI to flag calibration configs with same-vendor judge bias, weak held-out power, train/test leakage, or opaque routed-provider family risk.
- You need machine-readable `AnalyticalFinding` output that can feed `derive_adaptation_plan`.

## Verification Loop

```bash
python scripts/generate_readme_claims.py --check
python scripts/check_repo_consistency.py
python examples/demo_replay.py
python scripts/run_golden_cases.py --check
python scripts/verify_fixture_integrity.py
```

These commands are no-network by design. They verify that public claims, generated docs, demo fixtures, golden cases, and artifact digests still match local source of truth.

## Deterministic Demo

```bash
python examples/demo_replay.py
```

The demo loads JSONL/JSON fixtures from `examples/demo_config/`, runs `mini-antemortem-cli check` in text and JSON modes, and compares the replay against `examples/_demo_output.txt`. It uses no API keys and makes no network calls.

## How Is This Different?

| Dimension | `mini-antemortem-cli` | `mini-omega-lock` | `antemortem-cli` | `omegaprompt` default path | Ad-hoc review prompts |
|---|---|---|---|---|---|
| Core role | Deterministic analytical preflight over calibration config. | Empirical preflight over live or mocked provider behavior. | Broader pre-diff recon and implementation-risk CLI. | Calibration engine that consumes preflight outputs. | Free-form human/LLM review of a config. |
| Deterministic no-network behavior | Yes by default. | Mock mode can be deterministic; live mode is provider-dependent. | Not the default when provider recon is enabled. | Core calibration can call configured providers. | No guarantee. |
| Trap classification | Yes, over built-in calibration traps. | Measures endpoint/judge behavior rather than this static trap registry. | Can reason over broader risk lists. | Consumes `PreflightReport`; does not ship this classifier. | Prompt-dependent. |
| Explicit trap IDs | Yes: each finding has a stable `trap_id`. | Not this trap ID registry. | Uses its own evidence/recon structures. | Preserves supplied analytical findings. | Usually absent unless manually requested. |
| Source-backed trap count | Yes, generated from `analytical_traps()`. | Not applicable to this trap registry. | Not applicable to this mini package. | Not applicable. | No. |
| Train/test split discipline | Flags missing held-out slice and train/test ID overlap. | Can probe empirical behavior but does not replace split integrity checks. | Can inspect source/artifacts when configured. | Uses whatever datasets caller supplies. | Usually easy to miss. |
| Routed-provider opacity | Flags routed-provider family ambiguity as `UNRESOLVED`. | Can probe actual endpoint behavior when live calls are allowed. | Can gather external evidence when configured. | Does not infer provider family. | Often hidden by provider labels. |
| Same-vendor judge bias | Flags same-family target/judge pairs. | Can measure judge consistency but does not make this static config claim. | Can analyze broader judge-risk context. | Consumes findings if provided. | Often subjective. |
| CLI/MCP availability | CLI: `mini-antemortem-cli`; MCP: `mini-antemortem-cli-mcp` via `[mcp]`. | Separate sibling package. | Separate broader CLI. | Library API. | None unless built by the user. |
| Reads source files | No. It reads calibration input files only. | No by default. | Yes, for disk-backed recon and citations. | No source recon by default. | Only if pasted or tool-enabled. |
| Live empirical probes | No. | Yes in live mode. | Yes when configured. | Provider calls during calibration. | Maybe, but not reproducible by default. |
| Disk-verified file:line citations | No. Fixture integrity only. | No. | Yes, where that tool implements evidence-bound citations. | No. | No. |
| What it does not prove | It does not prove provider quality, prompt superiority, statistical validity, production adoption, or external validation. | It does not prove analytical trap absence. | It does not prove this mini package's trap count. | It does not perform this preflight unless supplied. | It proves nothing mechanically. |

## Built-In Trap Patterns

Source of truth: `src/mini_antemortem_cli/traps.py` via `analytical_traps()`.

| Trap ID | What it checks |
|---|---|
| `self_agreement_bias` | Target and judge share vendor family or exact model, creating self-agreement risk. |
| `small_sample_kc4_power` | Held-out sample size is too small for KC-4/Pearson signal to carry useful power. |
| `variants_homogeneous` | Prompt variants are too similar to create meaningful sensitivity signal. |
| `rubric_weight_concentration` | One rubric dimension dominates the weighted fitness. |
| `judge_budget_too_small` | SMALL judge output budget is likely insufficient for rubric dimensions and gates. |
| `empty_reference_with_strict_rubric` | Rubric implies ground-truth comparison but dataset references are absent. |
| `no_held_out_slice` | No test slice is provided, so walk-forward validation cannot run. |
| `train_test_id_overlap` | Train/test IDs overlap or duplicate IDs make per-item correlation unreliable. |
| `routed_provider_opaque_family` | A routed provider obscures the underlying served-model family. |

Each finding is one of `REAL`, `GHOST`, `NEW`, or `UNRESOLVED` and carries severity `blocker`, `high`, `medium`, or `low`.

## CLI

```bash
mini-antemortem-cli list-traps
mini-antemortem-cli check \
  --target-provider openai \
  --target-model gpt-4o \
  --judge-provider anthropic \
  --judge-model claude-opus-4-7 \
  --train examples/demo_config/train.jsonl \
  --test examples/demo_config/test.jsonl \
  --rubric examples/demo_config/rubric.json \
  --variants examples/demo_config/variants.json \
  --judge-output-budget small
```

Use `--json` for machine-readable output. Use `--fail-on-severity high` when CI should fail on high-or-worse `REAL`/`UNRESOLVED` findings. The deprecated `--fail-on-blocker` alias remains for backward compatibility.

## Python API

```python
from mini_antemortem_cli import analytical_preflight, analytical_traps
```

`analytical_preflight(...)` returns `omegaprompt.preflight.contracts.AnalyticalFinding` objects. The output is compatible with `omegaprompt.preflight.PreflightReport` and `derive_adaptation_plan`.

## MCP

```bash
pip install "mini-antemortem-cli[mcp]"
mini-antemortem-cli-mcp
# or
python -m mini_antemortem_cli.mcp
```

The MCP server exposes `analytical_preflight` and `list_traps`. Path inputs are bounded by `MINI_ANTEMORTEM_WORKSPACE_ROOT` or the current working directory; inline JSON objects do not touch the filesystem.

## Release Hygiene

```bash
python scripts/release_audit.py --no-network
python -m build
python scripts/wheel_smoke_install.py dist/*.whl
python scripts/publish_readiness.py --no-network
```

These scripts do not publish, tag, or create GitHub releases. Publishing is only wired through [.github/workflows/publish.yml](.github/workflows/publish.yml) on `v*.*.*` tags or manual dispatch, using PyPI Trusted Publishing / GitHub OIDC with no token secret. Setup and sequencing are documented in [docs/release_checklist.md](docs/release_checklist.md).

## License

Apache 2.0. See [LICENSE](LICENSE).

License history: PyPI distributions of version 0.1.0 were shipped with an MIT `LICENSE` file. The repository was relicensed to Apache 2.0 on 2026-04-22 (commit `d2d7eb7`); 0.2.0 and later versions ship under Apache 2.0. Anyone who installed 0.1.0 holds an MIT license to that copy; license changes do not apply retroactively.
