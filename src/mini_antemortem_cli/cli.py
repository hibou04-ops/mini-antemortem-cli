# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""``mini-antemortem-cli`` — terminal entrypoint for the analytical preflight.

Reviewer 3순위: the package was named ``mini-antemortem-cli`` but only
shipped an MCP entrypoint, so a user typing ``mini-antemortem-cli --help``
got nothing. This module ships a real CLI that loads the calibration
inputs from disk, runs ``analytical_preflight``, and prints findings as
text (default) or JSON.

Usage::

    mini-antemortem-cli check \\
      --target-provider openai \\
      --target-model gpt-4o \\
      --judge-provider anthropic \\
      --judge-model claude-opus-4-7 \\
      --train train.jsonl \\
      --test test.jsonl \\
      --rubric rubric.json \\
      --variants variants.json \\
      --judge-output-budget small

Add ``--json`` for machine-readable output (one ``AnalyticalFinding``
per row in the ``findings`` array). Exit code is 0 by default — the CLI
is non-blocking because analytical preflight is advisory, not a ship
gate — unless ``--fail-on-severity`` (or the deprecated
``--fail-on-blocker``) turns it into a policy gate, in which case a
matching finding exits 1. A configuration error (a missing or malformed
input file) exits 2. See ``docs/cli_exit_codes.md``.

The intent is to keep this CLI dependency-light (stdlib argparse +
pydantic, both already required by ``omegaprompt``) so it runs anywhere
``omegaprompt`` already runs — no extra installs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from omegaprompt.domain.dataset import Dataset
from omegaprompt.domain.judge import JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.preflight.contracts import AnalyticalFinding, PreflightSeverity

from mini_antemortem_cli import __version__
from mini_antemortem_cli.traps import (
    TrapPolicy,
    analytical_preflight,
    analytical_traps,
    summarize_findings,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-antemortem-cli",
        description=(
            "Analytical preflight for omegaprompt calibration. "
            f"Classifies {len(analytical_traps())} calibration trap patterns "
            "deterministically — no API calls, no network."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"mini-antemortem-cli {__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "check",
        help=f"Classify a calibration config against the {len(analytical_traps())} traps.",
    )
    check.add_argument("--target-provider", required=True)
    check.add_argument("--target-model", default=None)
    check.add_argument("--judge-provider", required=True)
    check.add_argument("--judge-model", default=None)
    check.add_argument(
        "--train",
        required=True,
        type=Path,
        help="Path to training dataset JSONL.",
    )
    check.add_argument(
        "--test",
        type=Path,
        default=None,
        help="Optional path to held-out test dataset JSONL.",
    )
    check.add_argument(
        "--rubric",
        required=True,
        type=Path,
        help="Path to JudgeRubric JSON.",
    )
    check.add_argument(
        "--variants",
        required=True,
        type=Path,
        help="Path to PromptVariants JSON.",
    )
    check.add_argument(
        "--judge-output-budget",
        default="small",
        help="LLM judge output budget bucket (small | medium | large). Default: small.",
    )
    check.add_argument(
        "--policy",
        type=Path,
        default=None,
        metavar="POLICY.JSON",
        help=(
            "Optional TrapPolicy JSON file overriding default thresholds "
            "(min_test_items_high, near_duplicate_jaccard, etc.). Field names "
            "match TrapPolicy dataclass; unknown fields are ignored."
        ),
    )
    check.add_argument(
        "--json",
        action="store_true",
        help="Emit findings as JSON to stdout instead of human-readable text.",
    )
    check.add_argument(
        "--fail-on-severity",
        choices=["low", "medium", "high", "blocker"],
        default=None,
        metavar="LEVEL",
        help=(
            "Exit non-zero when any finding's severity is at least LEVEL. "
            "Recommended for CI: --fail-on-severity high. Off by default — "
            "analytical preflight is advisory unless this flag is set."
        ),
    )
    check.add_argument(
        "--fail-on-label",
        default=None,
        metavar="LABELS",
        help=(
            "Comma-separated finding labels that trigger non-zero exit when "
            "combined with --fail-on-severity. Default: REAL,UNRESOLVED. "
            "Set to 'REAL' for stricter gates or 'REAL,NEW,UNRESOLVED' for "
            "broader coverage."
        ),
    )
    check.add_argument(
        "--fail-on-blocker",
        action="store_true",
        help=(
            "Deprecated alias for `--fail-on-severity blocker --fail-on-label "
            "REAL,NEW,UNRESOLVED`. As of 0.9.0 the train_test_id_overlap trap "
            "emits BLOCKER on exact train/test ID overlap, so this gate now "
            "trips on a real failure. Prefer --fail-on-severity high, which "
            "catches BLOCKER and HIGH alike."
        ),
    )

    list_traps = sub.add_parser(
        "list-traps",
        help=f"List the {len(analytical_traps())} built-in trap patterns and exit.",
    )
    list_traps.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the trap registry as a JSON array of {id, hypothesis} "
            "objects instead of human-readable text."
        ),
    )

    return parser


_SEVERITY_ORDER: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "blocker": 4,
}


def _should_fail(
    finding: AnalyticalFinding,
    *,
    min_severity: str,
    labels: set[str],
) -> bool:
    sev = str(getattr(finding.severity, "value", finding.severity)).lower()
    return (
        finding.label in labels
        and _SEVERITY_ORDER.get(sev, 0) >= _SEVERITY_ORDER[min_severity]
    )


def _load_variants(path: Path) -> PromptVariants:
    return PromptVariants.model_validate_json(path.read_text(encoding="utf-8"))


# Exit code 2 == configuration/usage error (matches argparse and
# docs/cli_exit_codes.md). Distinct from exit 1, which is a policy-gate
# failure (a finding met --fail-on-severity). A bad input file is not a
# policy decision, so it must not be confused with a gate trip.
_CONFIG_ERROR_EXIT = 2


class _InputLoadError(Exception):
    """A calibration input file could not be loaded (config error -> exit 2)."""


def _load_input(label: str, path: Path, loader):  # type: ignore[no-untyped-def]
    """Run ``loader(path)``; translate load failures into a structured error.

    Wraps each on-disk loader so a bad file produces a one-line stderr
    message naming the file and the error class — not a raw Python
    traceback. The path is printed exactly as the user supplied it
    (no ``.resolve()``), so absolute home-directory paths are never
    leaked into CI logs.

    Caught classes (verified empirically against the omegaprompt
    loaders, 2026-06-08):

    - ``FileNotFoundError`` — missing file (all loaders).
    - ``json.JSONDecodeError`` — malformed JSON (rubric ``from_json``).
    - ``ValidationError`` — schema mismatch / malformed JSON (pydantic
      ``model_validate_json`` path: variants).
    - ``ValueError`` — malformed JSON or schema mismatch wrapped by
      ``Dataset.from_jsonl`` (it re-raises both as ``ValueError``).
      ``json.JSONDecodeError`` subclasses ``ValueError``; the isinstance
      ladder labels it first so the message stays specific.
    """
    try:
        return loader(path)
    except FileNotFoundError as exc:
        raise _InputLoadError(
            f"{label} file not found: {path} (FileNotFoundError)"
        ) from exc
    except json.JSONDecodeError as exc:
        raise _InputLoadError(
            f"{label} file is not valid JSON: {path} (JSONDecodeError: {exc.msg})"
        ) from exc
    except ValidationError as exc:
        raise _InputLoadError(
            f"{label} file failed schema validation: {path} (ValidationError)"
        ) from exc
    except ValueError as exc:
        # Keep stderr to one line: Dataset.from_jsonl re-raises a bad-schema row as a
        # pydantic ValueError whose str() is a multi-line report. The docstring +
        # docs/cli_exit_codes.md promise a one-line message, so take the first line.
        detail = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        raise _InputLoadError(
            f"{label} file is malformed or schema-invalid: {path} (ValueError: {detail})"
        ) from exc


def _format_text(findings: Sequence[AnalyticalFinding]) -> str:
    # C1 (0.9.0): lead with one grep-friendly verdict line so the native
    # 5-level status (PASS/ADVISORY/HOLD/BLOCK/NEEDS_MORE_EVIDENCE) — the
    # whole point of the tool — is visible to the default (text) user, not
    # only to --json consumers. `... check | head -1` becomes the CI signal.
    summary = summarize_findings(list(findings))
    counts = summary["counts"]
    summary_line = (
        f"Summary: {summary['status']} [{summary['highest_severity']}] - "
        f"{counts['REAL']} REAL, {counts['GHOST']} GHOST, "
        f"{counts['NEW']} NEW, {counts['UNRESOLVED']} UNRESOLVED"
    )
    lines: list[str] = [summary_line, ""]
    severity_marker = {
        PreflightSeverity.BLOCKER: "[BLOCKER]",
        PreflightSeverity.HIGH: "[HIGH]   ",
        PreflightSeverity.MEDIUM: "[MEDIUM] ",
        PreflightSeverity.LOW: "[LOW]    ",
    }
    for f in findings:
        marker = severity_marker.get(f.severity, "[?]")
        lines.append(f"{marker} {f.label:<10} {f.trap_id}")
        if f.note:
            lines.append(f"            {f.note}")
        if f.remediation and f.label in {"REAL", "NEW"}:
            lines.append(f"            -> {f.remediation}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_json(findings: Sequence[AnalyticalFinding]) -> str:
    summary = summarize_findings(list(findings))
    return json.dumps(
        {
            **summary,
            "findings": [f.model_dump(mode="json") for f in findings],
        },
        indent=2,
        ensure_ascii=False,
    )


def _run_check(args: argparse.Namespace) -> int:
    try:
        train = _load_input("train dataset", args.train, Dataset.from_jsonl)
        test = (
            _load_input("test dataset", args.test, Dataset.from_jsonl)
            if args.test
            else None
        )
        rubric = _load_input("rubric", args.rubric, JudgeRubric.from_json)
        variants = _load_input("variants", args.variants, _load_variants)
        policy = (
            _load_input("policy", args.policy, TrapPolicy.from_json_file)
            if args.policy
            else None
        )
    except _InputLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _CONFIG_ERROR_EXIT

    findings = analytical_preflight(
        target_provider=args.target_provider,
        target_model=args.target_model,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        train_dataset=train,
        test_dataset=test,
        rubric=rubric,
        variants=variants,
        judge_output_budget=args.judge_output_budget,
        policy=policy,
    )

    if args.json:
        print(_format_json(findings))
    else:
        print(_format_text(findings))

    # Resolve gate. New flags win over the deprecated --fail-on-blocker.
    min_severity: str | None = args.fail_on_severity
    if min_severity is None and args.fail_on_blocker:
        min_severity = "blocker"
    if min_severity is not None:
        if args.fail_on_label:
            label_set = {tok.strip().upper() for tok in args.fail_on_label.split(",") if tok.strip()}
        else:
            label_set = {"REAL", "UNRESOLVED"}
        if any(_should_fail(f, min_severity=min_severity, labels=label_set) for f in findings):
            return 1
    return 0


def _run_list_traps(args: argparse.Namespace) -> int:
    if getattr(args, "json", False):
        # H1 (0.9.0): machine-readable registry for agent/CI consumers.
        # Text stays the default so the golden cli_list_traps.ids case
        # (which parses text line-by-line) is unaffected.
        print(
            json.dumps(
                [{"id": trap.id, "hypothesis": trap.hypothesis} for trap in analytical_traps()],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    for trap in analytical_traps():
        print(f"{trap.id}")
        print(f"  {trap.hypothesis}")
        print()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "check":
        return _run_check(args)
    if args.command == "list-traps":
        return _run_list_traps(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
