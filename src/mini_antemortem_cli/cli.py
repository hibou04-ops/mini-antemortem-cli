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
per row in the ``findings`` array). Exit code is 0 unless any finding
has ``severity=blocker``; the CLI is non-blocking by default because
analytical preflight is advisory, not a ship gate.

The intent is to keep this CLI dependency-free (stdlib argparse) so it
runs anywhere ``omegaprompt`` already runs — no extra installs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

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
            "Classifies seven calibration trap patterns deterministically — "
            "no API calls, no network."
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
        help="Classify a calibration config against the seven traps.",
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
            "REAL,NEW,UNRESOLVED`. Currently almost-always a no-op because no "
            "built-in trap emits BLOCKER severity. Use --fail-on-severity "
            "high for the CI gate users typically intend."
        ),
    )

    sub.add_parser(
        "list-traps",
        help="List the seven built-in trap patterns and exit.",
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


def _format_text(findings: Sequence[AnalyticalFinding]) -> str:
    lines: list[str] = []
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
    train = Dataset.from_jsonl(args.train)
    test = Dataset.from_jsonl(args.test) if args.test else None
    rubric = JudgeRubric.from_json(args.rubric)
    variants = _load_variants(args.variants)
    policy = TrapPolicy.from_json_file(args.policy) if args.policy else None

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


def _run_list_traps() -> int:
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
        return _run_list_traps()
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
