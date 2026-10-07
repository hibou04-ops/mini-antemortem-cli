# Changelog

## 0.10.1 - 2026-10-08

Bound FastMCP to SDK >=1,<2. Add an installed offline trap registry path and analytical findings handoff. Align README and release checklist with the existing manual workflow_dispatch/release_tag publish path.

Compatibility: no renamed imports, CLI/MCP identifiers, schemas or relaxed gates.
Upgrade with the same PyPI distribution name; MCP users reinstall its [mcp] extra.


All notable changes to `mini-antemortem-cli` are documented here. This
project adheres to [Semantic Versioning](https://semver.org/).

## [0.10.0] - 2026-06-12

This release adds two new trap rules (taking the source-backed count to
**eleven**), corrects a long-standing claim-drift in the package's own trap
count, makes the publish workflow version-agnostic, and ships a distribution
README overhaul.

### Added

- **`few_shot_leakage_into_test` trap.** Few-shot examples are baked into every
  prompt variant, so they are part of what the model sees at inference. When a
  few-shot example's `input` or `output` text matches a held-out test (or
  train) dataset item, the model has effectively been handed the answer — the
  score reflects memorisation, not generalisation. Fires `REAL`/`HIGH` on test
  overlap, `REAL`/`MEDIUM` on train-only overlap, and carries the leaked item
  ids in `cite`. This is distinct from `train_test_id_overlap`, which only
  compares item *ids* and never inspects few-shot content. Gated by the new
  `TrapPolicy.check_few_shot_leakage` flag (default `True`).
- **`rubric_dead_weight_dimension` trap.** `JudgeRubric` only requires the
  *sum* of dimension weights to be positive, so an individual dimension may
  legally carry `weight == 0.0`. Such a dimension is still serialised into the
  judge prompt verbatim (spending tokens and attention) but contributes nothing
  to the fitness `normalized_weights()` produces. Fires `REAL`/`MEDIUM` when a
  zero-weight dimension coexists with a weighted one; the dead dimension names
  are surfaced in `cite`.
- **`TrapPolicy.check_few_shot_leakage`.** New tuneable flag (default `True`)
  to disable the few-shot-leakage check for tasks where few-shot inputs are
  intentionally drawn from the eval distribution.

### Changed

- **Source-backed trap count: 9 → 11.** Both new traps are deterministic and
  reachable through the real `omegaprompt` domain objects. The false-positive
  corpus grew to 53 cases (0/53, 0.00%) with ≥3 benign cases per new trap, and
  golden cases cover the REAL and GHOST classification of each.

### Fixed

- **Claim-drift in the package's own trap count.** The package described its
  trap count inconsistently — `pyproject.toml` said "nine" while the GitHub
  repository description said "seven". The single source of truth is
  `analytical_traps()`; every reference (`pyproject`, all four READMEs, the
  `__init__` docstring, the MCP server instructions, and the generated claim
  docs) is now derived from it, the count is correct everywhere (11), and
  `scripts/check_repo_consistency.py` was hardened to also enforce trap-ID
  presence in `docs/trust_model.*` (which previously could drift silently).
- **Version-agnostic publish workflow.** `.github/workflows/publish.yml`
  previously used a hard-coded `v0.4.0` example tag while pyproject had drifted
  ahead. The workflow now adds a "Verify version agreement" step that reads the
  version from `pyproject.toml` via `tomllib`, parses `__init__.__version__`,
  strips the leading `v` from the release tag, and fails the build unless all
  three match — so a tag/metadata mismatch fails fast instead of silently
  publishing the wrong version.

## [0.9.1] - 2026-06-08

### Fixed

- **Config-load `ValueError` stderr is now a single line.** `Dataset.from_jsonl`
  re-raises a bad-schema row as a plain `ValueError` whose string is a multi-line
  pydantic report, so a bad-schema `--train` / `--test` file emitted a multi-line
  stderr — contradicting the documented one-line guarantee (the other loader
  branches were already terse). Only the first line is now emitted. The exit code
  is unchanged (still `2`); a regression test locks the single-line behavior.

## [0.9.0] - 2026-06-08

This release makes the verdict visible, makes findings self-contained, and
fails cleanly — then settles the one outstanding severity-contract question
and moves the project to Beta.

### Added

- **Text-mode verdict line (C1).** `check` text output now leads with one
  grep-friendly `Summary:` line surfacing the existing native 5-level status
  (`PASS` / `ADVISORY` / `HOLD` / `BLOCK` / `NEEDS_MORE_EVIDENCE`), label
  counts, and highest severity. The core verdict, previously visible only to
  `--json` consumers, is now visible to the default (text) user.
- **Config-referenced citations (C2).** High-signal findings now populate the
  existing optional `cite` field with the computed value the trap fired on:
  `train_test_id_overlap` (overlapping ids), `rubric_weight_concentration`
  (dominant dimension and weight), `variants_homogeneous` (max pairwise
  Jaccard), and `small_sample_kc4_power` (test slice size vs threshold).
  These reference the supplied calibration config, not on-disk source files;
  the `AnalyticalFinding` schema is unchanged.
- **`list-traps --json` (H1).** The `list-traps` subcommand accepts `--json`
  and emits a `[{"id", "hypothesis"}, ...]` array. Text remains the default.

### Changed

- **Exact train/test ID overlap is now `BLOCKER` (H2) — behavior change.**
  The `train_test_id_overlap` trap emits `blocker` severity (was `high`) on
  exact train/test ID overlap: the held-out set is not held out, a hard leak.
  Within-slice duplicates remain `medium`. This is a public severity-contract
  change. `--fail-on-severity high` still catches `blocker` (4 ≥ 3 in the
  severity order), so no CI gate regresses; the false-positive audit stays
  0/45 (the benign corpus has no overlap cases by construction). The
  `summarize_findings` roll-up now reports `BLOCK` status for an overlap.
- **Input-file load failures now exit cleanly (C3).** Missing files, malformed
  JSON, and schema mismatches in `--train` / `--test` / `--rubric` /
  `--variants` / `--policy` are caught and reported as a one-line stderr
  message naming the file (as the user supplied it) and the error class, then
  the CLI exits `2` (configuration error — distinct from the policy-gate exit
  `1`). Previously these propagated as raw Python tracebacks.
- **Development status: `3 - Alpha` -> `4 - Beta`.** Going forward, changes to
  the CLI / JSON / MCP surface are committed to be additive-only.
- Documentation: the README/`trust_model` "Disk-verified file:line citations:
  No" row now distinguishes disk citations (still No) from the new
  config-referenced citations (Yes, via C2). The `--fail-on-blocker` help text
  and exit-code docs reflect that overlap now emits `BLOCKER`.

[0.10.0]: https://github.com/hibou04-ops/mini-antemortem-cli/releases/tag/v0.10.0
[0.9.1]: https://github.com/hibou04-ops/mini-antemortem-cli/releases/tag/v0.9.1
[0.9.0]: https://github.com/hibou04-ops/mini-antemortem-cli/releases/tag/v0.9.0
