"""Audit analytical-preflight false-positive rate against benign-case corpus.

Replays every case in ``benchmarks/false_positive/benign_cases.json`` through
:func:`mini_antemortem_cli.traps.analytical_preflight` and counts how often the
named trap fires (label REAL / NEW / UNRESOLVED) on a configuration the corpus
labels as benign. A false positive is any benign case that fires.

Run ``python scripts/run_false_positive_audit.py`` for a human report,
``--json`` for the machine report, ``--check`` to exit 1 on any FP (CI gate).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mini_antemortem_cli.traps import analytical_preflight  # noqa: E402
from omegaprompt.domain.dataset import Dataset, DatasetItem  # noqa: E402
from omegaprompt.domain.judge import Dimension, HardGate, JudgeRubric  # noqa: E402
from omegaprompt.domain.params import PromptVariants  # noqa: E402


MANIFEST = ROOT / "benchmarks" / "false_positive" / "benign_cases.json"

# A finding is treated as "trap fired" when its label is one of these.
# GHOST is the only fully-silent label; NEW and UNRESOLVED still surface
# the trap to the user as something worth attention, so they count as
# positives for FP measurement.
POSITIVE_LABELS: frozenset[str] = frozenset({"REAL", "NEW", "UNRESOLVED"})


def _build_dataset(spec: dict[str, Any] | None) -> Dataset | None:
    if spec is None:
        return None
    prefix = spec["prefix"]
    n = spec["n"]
    with_ref = spec.get("with_ref", True)
    items = [
        DatasetItem(
            id=f"{prefix}{i:03d}",
            input=f"task {prefix}{i:03d}",
            reference=f"ref {prefix}{i:03d}" if with_ref else None,
        )
        for i in range(n)
    ]
    return Dataset(items=items)


def _build_rubric(spec: dict[str, Any]) -> JudgeRubric:
    weights = spec.get("weights") or {"accuracy": 0.5, "clarity": 0.5}
    needs_ref = spec.get("needs_reference", False)
    gates = spec.get("gates", 1)
    description = (
        "Matches the expected output." if needs_ref else "Self-contained quality score."
    )
    return JudgeRubric(
        dimensions=[
            Dimension(name=name, description=description, weight=weight)
            for name, weight in weights.items()
        ],
        hard_gates=[
            HardGate(name=f"gate_{i}", description="Must attempt the task.", evaluator="judge")
            for i in range(gates)
        ],
    )


def _build_variants(spec: dict[str, Any]) -> PromptVariants:
    return PromptVariants(
        system_prompts=spec["prompts"],
        few_shot_examples=[{"input": "1+1", "output": "2"}],
    )


def _merge(baseline: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Override merge: top-level keys in ``overrides`` replace baseline wholesale.

    Sub-dicts (rubric, train_dataset, ...) are replaced as a single unit,
    not deep-merged. This matches the manifest's contract that an override
    block describes the full spec for the keys it touches.
    """
    merged = dict(baseline)
    merged.update(overrides)
    return merged


def _build_kwargs(merged: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_provider": merged["target_provider"],
        "target_model": merged["target_model"],
        "judge_provider": merged["judge_provider"],
        "judge_model": merged["judge_model"],
        "train_dataset": _build_dataset(merged["train_dataset"]),
        "test_dataset": _build_dataset(merged.get("test_dataset")),
        "rubric": _build_rubric(merged["rubric"]),
        "variants": _build_variants(merged["variants"]),
        "judge_output_budget": merged["judge_output_budget"],
    }


def _sev(value: object) -> str:
    return str(getattr(value, "value", value)).lower()


def run_audit(manifest_path: Path = MANIFEST) -> dict[str, Any]:
    """Replay every benign case and return per-trap + overall FP stats."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline = manifest["baseline"]
    cases = manifest["cases"]

    per_trap_acc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "false_positives": 0, "false_positive_case_ids": []}
    )
    case_results: list[dict[str, Any]] = []

    for case in cases:
        merged = _merge(baseline, case.get("overrides") or {})
        findings = analytical_preflight(**_build_kwargs(merged))
        finding = next(f for f in findings if f.trap_id == case["trap_id"])
        is_fp = finding.label in POSITIVE_LABELS
        bucket = per_trap_acc[case["trap_id"]]
        bucket["total"] += 1
        if is_fp:
            bucket["false_positives"] += 1
            bucket["false_positive_case_ids"].append(case["case_id"])
        case_results.append(
            {
                "case_id": case["case_id"],
                "trap_id": case["trap_id"],
                "label": finding.label,
                "severity": _sev(finding.severity),
                "is_false_positive": is_fp,
            }
        )

    per_trap_out: dict[str, Any] = {}
    total_cases = 0
    total_fps = 0
    for trap_id, stats in per_trap_acc.items():
        n = stats["total"]
        fp = stats["false_positives"]
        per_trap_out[trap_id] = {
            "total_cases": n,
            "false_positives": fp,
            "false_positive_rate": (fp / n) if n else 0.0,
            "false_positive_case_ids": list(stats["false_positive_case_ids"]),
        }
        total_cases += n
        total_fps += fp

    acknowledged = manifest.get("acknowledged_false_positives") or []
    acknowledged_ids = {entry["case_id"] for entry in acknowledged}
    observed_fp_ids = {c["case_id"] for c in case_results if c["is_false_positive"]}
    unexpected_fp_ids = sorted(observed_fp_ids - acknowledged_ids)
    stale_acknowledgements = sorted(acknowledged_ids - observed_fp_ids)

    return {
        "manifest_schema": manifest.get("schema"),
        "total_cases": total_cases,
        "total_false_positives": total_fps,
        "overall_false_positive_rate": (total_fps / total_cases) if total_cases else 0.0,
        "per_trap": per_trap_out,
        "cases": case_results,
        "acknowledged_false_positives": list(acknowledged),
        "unexpected_false_positive_case_ids": unexpected_fp_ids,
        "stale_acknowledgement_case_ids": stale_acknowledgements,
    }


def _format_human(report: dict[str, Any]) -> str:
    ack_ids = {entry["case_id"] for entry in report.get("acknowledged_false_positives", [])}
    lines = [
        f"benign cases: {report['total_cases']}",
        f"false positives: {report['total_false_positives']} "
        f"(acknowledged {len(ack_ids)}, unexpected {len(report['unexpected_false_positive_case_ids'])})",
        f"overall FP rate: {report['overall_false_positive_rate']:.2%}",
        "per-trap breakdown:",
    ]
    for trap_id, stats in sorted(report["per_trap"].items()):
        marker = "OK" if stats["false_positives"] == 0 else "FP"
        lines.append(
            f"  [{marker}] {trap_id}: {stats['false_positives']}/{stats['total_cases']} "
            f"({stats['false_positive_rate']:.2%})"
        )
        for cid in stats["false_positive_case_ids"]:
            offending = next(c for c in report["cases"] if c["case_id"] == cid)
            tag = "ACK" if cid in ack_ids else "NEW_FP"
            lines.append(
                f"        - [{tag}] {cid} fired as {offending['label']}/{offending['severity']}"
            )
    if report["stale_acknowledgement_case_ids"]:
        lines.append(
            "stale acknowledgements (no longer firing — please remove): "
            + ", ".join(report["stale_acknowledgement_case_ids"])
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Exit 1 when an unexpected FP is observed (a benign case fires "
            "that is not in acknowledged_false_positives) OR a stale "
            "acknowledgement is present (no longer firing)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full JSON report on stdout instead of the human summary.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the JSON report to this path (in addition to stdout).",
    )
    args = parser.parse_args(argv)

    report = run_audit()
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_format_human(report))

    if args.check and (
        report["unexpected_false_positive_case_ids"]
        or report["stale_acknowledgement_case_ids"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
