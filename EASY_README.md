# mini-antemortem-cli — Easy Start

[English](README.md) · [한국어](README.ko.md) · [English easy](EASY_README.md) · [한국어 easy](EASY_README_KR.md)

A deterministic linter for 11 structural traps in calibration configuration: leakage, judge bias, homogeneous variants and related risks. It emits `AnalyticalFinding` records from local configuration without provider calls or network access.

Repository: `hibou04-ops/mini-antemortem-cli` · PyPI: `mini-antemortem-cli` · import: `mini_antemortem_cli` · CLI: `mini-antemortem-cli` · MCP: `mini-antemortem-cli-mcp` with `mini-antemortem-cli[mcp]`

## 0.11.3 — Optional bridge contract regression checks

The sole required dependency is `pydantic>=2.6.0`. Necessary types, exceptions and contracts live in the internal `_vendor` namespace, so import and CLI work without omegaprompt, omega-lock, NumPy or provider SDKs. `omegaprompt>=1.1.0` is confined to optional extra `[bridge]`. Local and parent models are separate Python classes.

Requires Python 3.11+. This checkout has source version **0.11.3**. The commands below install the checkout. These changes target a GitHub source release; PyPI publication uses a separate manual workflow. Publication of the new version on PyPI is not claimed.

```bash
python -m pip install .
mini-antemortem-cli --version
mini-antemortem-cli list-traps --json
python examples/demo_replay.py
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
