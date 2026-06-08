# Changelog

All notable changes to `mini-antemortem-cli` are documented here. This
project adheres to [Semantic Versioning](https://semver.org/).

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

[0.9.1]: https://github.com/hibou04-ops/mini-antemortem-cli/releases/tag/v0.9.1
[0.9.0]: https://github.com/hibou04-ops/mini-antemortem-cli/releases/tag/v0.9.0
