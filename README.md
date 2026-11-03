# mini-antemortem-cli

<!-- QUICKSTART_DEMO_START -->
## 10-second quickstart: reproduce a leakage gate

```bash
pip install mini-antemortem-cli
uvx mini-antemortem-cli --help
```

```diff
$ python -m mini_antemortem_cli.quickstart_demo
- [FAIL] overlapping train/test IDs: gate exit 1
+ [PASS] disjoint train/test IDs: gate exit 0
+ [PASS] checked 11 local traps per fixture
```

This synthetic fixture demo ships in the installed package. Its summary lines below are printed after checking real CLI JSON and exit codes. `check` itself emits the full findings; this demo does not measure model quality or generalization.
<!-- QUICKSTART_DEMO_END -->

[English](README.md) · [한국어](README.ko.md) · [English easy](EASY_README.md) · [한국어 easy](EASY_README_KR.md)

A deterministic linter for 11 structural traps in calibration configuration: leakage, judge bias, homogeneous variants and related risks. It emits `AnalyticalFinding` records from local configuration without provider calls or network access.

[![CI](https://github.com/hibou04-ops/mini-antemortem-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/hibou04-ops/mini-antemortem-cli/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mini-antemortem-cli.svg)](https://pypi.org/project/mini-antemortem-cli/)

Repository: `hibou04-ops/mini-antemortem-cli` · PyPI: `mini-antemortem-cli` · import: `mini_antemortem_cli` · CLI: `mini-antemortem-cli` · MCP: `mini-antemortem-cli-mcp` with `mini-antemortem-cli[mcp]`

## 0.11.3 — Optional bridge contract regression checks

The sole required dependency is `pydantic>=2.6.0`. Necessary types, exceptions and contracts live in the internal `_vendor` namespace, so import and CLI work without omegaprompt, omega-lock, NumPy or provider SDKs. `omegaprompt>=1.1.0` is confined to optional extra `[bridge]`. Local and parent models are separate Python classes.

Requires Python 3.11+. Package metadata and the release tag target **0.11.3**. The default installation uses PyPI; the optional bridge remains an explicit extra. For development from a checkout, use `python -m pip install -e ".[dev,mcp]"`.

```bash
pip install mini-antemortem-cli
mini-antemortem-cli --version
mini-antemortem-cli list-traps --json
python -m mini_antemortem_cli.quickstart_demo
```

The registry command and fixture demo need no API keys. Installation may download dependencies; the checks themselves are offline. CLI `check` can return 0 with serious findings in advisory mode. Specify `--fail-on-severity high` for CI policy: a policy breach returns 1; invalid configuration returns 2.

## Trap registry

Source: `src/mini_antemortem_cli/traps.py`, `analytical_traps()`. Existing IDs and stable codes `OMG001` through `OMG011` remain. `get_canonical_trap_code()` raises `KeyError` for unknown names.

| Trap ID | Check |
|---|---|
| `self_agreement_bias` | Target and judge share a vendor family or model. |
| `small_sample_kc4_power` | Held-out sample size is below the configured threshold. |
| `variants_homogeneous` | Prompt variants have high token-set similarity. |
| `rubric_weight_concentration` | One rubric dimension dominates the total weight. |
| `judge_budget_too_small` | The judge output budget is small for the rubric. |
| `empty_reference_with_strict_rubric` | A reference-dependent rubric has missing references. |
| `no_held_out_slice` | No test slice is supplied. |
| `train_test_id_overlap` | Train/test IDs overlap or IDs are duplicated. |
| `routed_provider_opaque_family` | A routed provider obscures the underlying model family. |
| `few_shot_leakage_into_test` | Few-shot inputs match evaluation inputs; pool exposure and selected exposure are distinguished. |
| `rubric_dead_weight_dimension` | A zero-weight dimension is still sent to the judge. |

Finding states are `REAL`, `GHOST`, `NEW`, `UNRESOLVED`; severities are `blocker`, `high`, `medium`, `low`. Registry or lint success does not establish hold-out generalization, model quality or deployment authorization.

## Input and policy contracts

Few-shot leakage checks input overlap after Unicode NFC and whitespace normalization while preserving case. Shared output labels for different inputs do not establish leakage. Without `selected_few_shot_indices`/`--selected-few-shot-indices`, checks describe potential pool exposure; with them, checks cover selected inputs.

`validation_mode` is `auto`, `paired` or `disjoint`. In `paired` mode, matching ID sets, inputs and references identify repeated measurement as `UNRESOLVED/MEDIUM`; partial overlap or conflicts remain `REAL/BLOCKER`. `TrapPolicy.fail_on_missing_test=False` changes the missing-test finding policy; it does not perform hold-out measurement. Unknown policy keys, invalid types, NaN/Inf and out-of-range thresholds are input errors.

## CLI / Python / MCP

```bash
mini-antemortem-cli check \
  --target-provider openai --target-model gpt-4o \
  --judge-provider anthropic --judge-model claude-opus-4-7 \
  --train examples/demo_config/train.jsonl \
  --test examples/demo_config/test.jsonl \
  --rubric examples/demo_config/rubric.json \
  --variants examples/demo_config/variants.json \
  --judge-output-budget small --fail-on-severity high --json
```

```python
from mini_antemortem_cli import analytical_preflight, analytical_traps

print([trap.id for trap in analytical_traps()])
```

`analytical_preflight(...)` returns local finding models. For parent integration, install `mini-antemortem-cli[bridge]`, explicitly convert with `mini_antemortem_cli.bridge.to_omegaprompt_findings(findings)`, then supply the result to the parent's `PreflightReport`. Conversion validates JSON fields against the installed parent models. Core imports do not eagerly load the parent.

```bash
python -m pip install ".[mcp]"
mini-antemortem-cli-mcp
```

MCP tools are `analytical_preflight` and `list_traps`. Supported MCP SDK range is `>=1.0.0,<2.0.0`; 2.x runtime support is not claimed. Path inputs are bounded by `MINI_ANTEMORTEM_WORKSPACE_ROOT` or the current directory. Inline JSON does not read files.

## Local verification

```bash
python -m pytest
python scripts/generate_readme_claims.py --check
python scripts/check_repo_consistency.py
python scripts/run_golden_cases.py --check
python scripts/run_false_positive_audit.py --check
python scripts/verify_fixture_integrity.py
```

These checks validate local contracts and fixture regressions. They do not establish utility on external OOS datasets or a false-positive rate across all configurations. Vendored source attribution and licenses are retained in [NOTICE](NOTICE) and `src/mini_antemortem_cli/_vendor/`.

## Optional bridge contract checks

Tests compare field and validation schemas plus Enum values between vendored models and the actual installed `omegaprompt`, then roundtrip JSON in both directions. Invalid inputs must retain matching error types and locations. These checks make no live provider calls.

```bash
python -m pip install ".[dev,mcp,bridge]"
python -m pytest -m bridge tests/test_bridge_contracts.py
```

Only an absent parent skips this optional integration file. Broken imports or changed contracts in an installed parent fail. Standalone core tests still run without the parent; required dependencies are unchanged.

## Evidence and documentation

- Generated claims: [English](docs/generated/claims.md) / [한국어](docs/generated/claims_kr.md)
- Trust model: [English](docs/trust_model.md) / [한국어](docs/trust_model_kr.md)
- Toolkit positioning: [English](docs/toolkit_positioning.md) / [한국어](docs/toolkit_positioning_kr.md)
- Claim ledger: [English](docs/claim_ledger.md) / [한국어](docs/claim_ledger_kr.md)
- Examples: [English](docs/examples.md) / [한국어](docs/examples_kr.md)
- [CLI exit codes](docs/cli_exit_codes.md) · [Release checklist](docs/release_checklist.md) · [Changelog](CHANGELOG.md)

[README.ko.md](README.ko.md) is the canonical Korean document; [README_KR.md](README_KR.md) is an identical compatibility copy.

Apache 2.0. [LICENSE](LICENSE) · [NOTICE](NOTICE).
