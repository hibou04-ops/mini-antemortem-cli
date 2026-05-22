"""Run deterministic offline golden cases."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mini_antemortem_cli.cli import main as cli_main  # noqa: E402
from mini_antemortem_cli.traps import TrapPolicy, analytical_preflight, analytical_traps  # noqa: E402
from omegaprompt.domain.dataset import Dataset, DatasetItem  # noqa: E402
from omegaprompt.domain.judge import Dimension, HardGate, JudgeRubric  # noqa: E402
from omegaprompt.domain.params import PromptVariants  # noqa: E402


EXPECTED = ROOT / "benchmarks" / "golden_cases" / "expected_cases.json"


def _dataset(prefix: str, n: int, *, with_ref: bool = True, ids: list[str] | None = None) -> Dataset:
    item_ids = ids or [f"{prefix}{i:03d}" for i in range(n)]
    return Dataset(
        items=[
            DatasetItem(
                id=item_id,
                input=f"task {item_id}",
                reference=f"ref {item_id}" if with_ref else None,
            )
            for item_id in item_ids
        ]
    )


def _rubric(weights: dict[str, float] | None = None, *, needs_reference: bool = False, gates: int = 1) -> JudgeRubric:
    weights = weights or {"accuracy": 0.5, "clarity": 0.5}
    description = "Matches the expected output." if needs_reference else "Self-contained quality score."
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


def _variants(prompts: list[str] | None = None) -> PromptVariants:
    return PromptVariants(
        system_prompts=prompts
        or [
            "Answer with concise arithmetic reasoning.",
            "Explain assumptions before solving the task.",
            "Return a direct final answer after checking edge cases.",
            "Use a formal rubric before writing the response.",
        ],
        few_shot_examples=[{"input": "1+1", "output": "2"}],
    )


def _base_kwargs() -> dict[str, Any]:
    return {
        "target_provider": "openai",
        "target_model": "gpt-4o",
        "judge_provider": "anthropic",
        "judge_model": "claude-opus-4-7",
        "train_dataset": _dataset("t", 20),
        "test_dataset": _dataset("v", 20),
        "rubric": _rubric(),
        "variants": _variants(),
        "judge_output_budget": "medium",
    }


def _sev(value: object) -> str:
    return str(getattr(value, "value", value)).lower()


def _preflight_case(trap_id: str, **overrides: Any) -> dict[str, Any]:
    kwargs = _base_kwargs()
    kwargs.update(overrides)
    findings = analytical_preflight(**kwargs)
    finding = next(f for f in findings if f.trap_id == trap_id)
    return {
        "trap_id": trap_id,
        "label": finding.label,
        "severity": _sev(finding.severity),
    }


def _run_cli(args: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli_main(args)
    return rc, buf.getvalue()


def _demo_cli_args() -> list[str]:
    config = ROOT / "examples" / "demo_config"
    return [
        "check",
        "--target-provider",
        "openai",
        "--target-model",
        "gpt-4o",
        "--judge-provider",
        "openai",
        "--judge-model",
        "gpt-4o",
        "--train",
        str(config / "train.jsonl"),
        "--test",
        str(config / "test.jsonl"),
        "--rubric",
        str(config / "rubric.json"),
        "--variants",
        str(config / "variants.json"),
        "--judge-output-budget",
        "small",
    ]


def _cli_json_demo() -> dict[str, Any]:
    rc, stdout = _run_cli([*_demo_cli_args(), "--json"])
    payload = json.loads(stdout)
    return {"return_code": rc, "status": payload["status"], "trap_count": len(payload["findings"])}


def _cli_fail_high() -> dict[str, Any]:
    rc, _ = _run_cli([*_demo_cli_args(), "--fail-on-severity", "high"])
    return {"return_code": rc}


def _cli_list_traps() -> dict[str, Any]:
    rc, stdout = _run_cli(["list-traps"])
    ids = {line.strip() for line in stdout.splitlines() if line.strip()}
    trap_ids = {trap.id for trap in analytical_traps()}
    return {"return_code": rc, "trap_count": len(ids & trap_ids)}


CASE_BUILDERS: dict[str, Callable[[], dict[str, Any]]] = {
    "self_agreement_bias.real_high": lambda: _preflight_case(
        "self_agreement_bias",
        judge_provider="openai",
        judge_model="gpt-4o",
    ),
    "self_agreement_bias.ghost_cross_vendor": lambda: _preflight_case("self_agreement_bias"),
    "small_sample_kc4_power.real_high": lambda: _preflight_case(
        "small_sample_kc4_power",
        train_dataset=_dataset("t", 12),
        test_dataset=_dataset("v", 8),
    ),
    "small_sample_kc4_power.ghost_adequate": lambda: _preflight_case("small_sample_kc4_power"),
    "variants_homogeneous.real_single_prompt": lambda: _preflight_case(
        "variants_homogeneous",
        variants=_variants(["Only one prompt."]),
    ),
    "variants_homogeneous.ghost_diverse": lambda: _preflight_case("variants_homogeneous"),
    "rubric_weight_concentration.real_medium": lambda: _preflight_case(
        "rubric_weight_concentration",
        rubric=_rubric({"accuracy": 0.9, "clarity": 0.1}),
    ),
    "rubric_weight_concentration.ghost_balanced": lambda: _preflight_case("rubric_weight_concentration"),
    "judge_budget_too_small.real_medium": lambda: _preflight_case(
        "judge_budget_too_small",
        rubric=_rubric({f"d{i}": 1.0 / 6 for i in range(6)}, gates=2),
        judge_output_budget="small",
    ),
    "judge_budget_too_small.ghost_medium_budget": lambda: _preflight_case("judge_budget_too_small"),
    "empty_reference_with_strict_rubric.real_medium": lambda: _preflight_case(
        "empty_reference_with_strict_rubric",
        train_dataset=_dataset("t", 20, with_ref=False),
        rubric=_rubric({"accuracy": 1.0}, needs_reference=True),
    ),
    "empty_reference_with_strict_rubric.ghost_refs_present": lambda: _preflight_case(
        "empty_reference_with_strict_rubric",
        rubric=_rubric({"accuracy": 1.0}, needs_reference=True),
    ),
    "no_held_out_slice.real_high": lambda: _preflight_case("no_held_out_slice", test_dataset=None),
    "no_held_out_slice.ghost_present": lambda: _preflight_case("no_held_out_slice"),
    "train_test_id_overlap.real_high": lambda: _preflight_case(
        "train_test_id_overlap",
        train_dataset=_dataset("t", 10, ids=["shared", *[f"t{i}" for i in range(9)]]),
        test_dataset=_dataset("v", 10, ids=["shared", *[f"v{i}" for i in range(9)]]),
    ),
    "train_test_id_overlap.ghost_disjoint": lambda: _preflight_case("train_test_id_overlap"),
    "routed_provider_opaque_family.unresolved_openrouter": lambda: _preflight_case(
        "routed_provider_opaque_family",
        target_provider="openrouter",
    ),
    "routed_provider_opaque_family.ghost_first_party": lambda: _preflight_case("routed_provider_opaque_family"),
    "policy_override.sample_power_relaxed": lambda: _preflight_case(
        "small_sample_kc4_power",
        train_dataset=_dataset("t", 12),
        test_dataset=_dataset("v", 8),
        policy=TrapPolicy(min_test_items_high=5, min_total_items_medium=20),
    ),
    "cli_json.demo_fixture": _cli_json_demo,
    "cli_fail_on_severity.high": _cli_fail_high,
    "cli_list_traps.ids": _cli_list_traps,
}


def load_expected() -> dict[str, Any]:
    return json.loads(EXPECTED.read_text(encoding="utf-8"))


def run_checks() -> list[str]:
    spec = load_expected()
    errors: list[str] = []
    seen_traps = set()

    for case in spec["cases"]:
        case_id = case["case_id"]
        builder = CASE_BUILDERS.get(case_id)
        if builder is None:
            errors.append(f"missing builder for {case_id}")
            continue
        observed = builder()
        expected = case["expected"]
        for key, expected_value in expected.items():
            if observed.get(key) != expected_value:
                errors.append(
                    f"{case_id}: expected {key}={expected_value!r}, got {observed.get(key)!r}"
                )
        if case.get("trap_id"):
            seen_traps.add(case["trap_id"])

    missing = {trap.id for trap in analytical_traps()} - seen_traps
    if missing:
        errors.append(f"golden cases missing trap coverage: {sorted(missing)}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail on mismatch.")
    args = parser.parse_args(argv)
    errors = run_checks()
    if errors:
        for error in errors:
            print(f"GOLDEN_CASE_FAIL: {error}", file=sys.stderr)
        return 1
    print("GOLDEN_CASES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
