# Trust Model

`mini-antemortem-cli` verifies structural properties of an `omegaprompt`
calibration configuration. It does not verify model quality.

## What It Verifies

- The source-backed built-in trap registry exposed by `analytical_traps()`.
- Deterministic classification of local dataset/rubric/variant inputs.
- Explicit trap IDs on every `AnalyticalFinding`.
- Config-referenced citations (`cite` field), as of 0.9.0, on the
  high-signal traps: the computed value the trap fired on (overlapping
  ids, dominant rubric dimension and weight, max pairwise variant
  Jaccard, undersized test slice). These reference the supplied
  calibration config, not on-disk source files.
- Severity assignment according to local classifier rules.
- CLI policy gates via `--fail-on-severity` and `--fail-on-label`.
- MCP path boundaries for filesystem inputs.
- Compatibility shape with `omegaprompt.preflight.PreflightReport` and `derive_adaptation_plan`.

Current trap IDs:

- `self_agreement_bias`
- `small_sample_kc4_power`
- `variants_homogeneous`
- `rubric_weight_concentration`
- `judge_budget_too_small`
- `empty_reference_with_strict_rubric`
- `no_held_out_slice`
- `train_test_id_overlap`
- `routed_provider_opaque_family`

## What It Does Not Verify

- Provider/model superiority.
- Statistical validity of a calibration result.
- Production adoption or external validation.
- Live endpoint reliability.
- Disk-verified source file citations. (The 0.9.0 `cite` field references
  the supplied calibration config, not on-disk source files.)
- Append-only audit trails.

## Deterministic / No-Network Boundary

Default tests, demo replay, golden cases, fixture integrity, release audit, and
publish readiness are designed to run without provider calls or network access.
The package reads local files or inline objects only.

## Trap Policy Threshold Boundary

`TrapPolicy` controls thresholds such as minimum held-out size, near-duplicate
Jaccard, rubric concentration, and small-budget axis limit. A policy override
changes classification thresholds; it does not change the trap registry.

## Train/Test Split Caveats

The package can flag a missing held-out slice and train/test ID overlap. It
cannot prove the held-out slice is representative, independent, or large enough
for every domain.

## Same-Vendor And Routed-Provider Limitations

Same-family target/judge pairs are flagged as self-agreement risk. Routed
providers are flagged as `UNRESOLVED` because the served-model family is not
visible to a static config pass.

## Severity Semantics

Severity is a local policy signal:

- `blocker`: hard stop if emitted.
- `high`: strong enough for CI gates and adaptation-plan discipline.
- `medium`: actionable warning.
- `low`: informational or clean-path evidence.

## CLI Fail-On Policy

By default, `check` exits `0` after reporting findings. `--fail-on-severity`
turns the CLI into a policy gate for labels selected by `--fail-on-label`
(`REAL,UNRESOLVED` by default).

## MCP Boundary

The MCP server exposes `analytical_preflight` and `list_traps`. Path inputs are
restricted to `MINI_ANTEMORTEM_WORKSPACE_ROOT` or the current working directory.
Inline dict/list inputs bypass filesystem reads.

## Adaptation-Plan Compatibility

Findings are emitted as `omegaprompt.preflight.contracts.AnalyticalFinding`
objects. This package does not mutate an `AdaptationPlan`; it supplies inputs
that `omegaprompt` can consume.
