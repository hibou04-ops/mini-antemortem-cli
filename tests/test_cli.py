"""Reviewer 3순위: real CLI surface for mini-antemortem-cli.

The package was named ``mini-antemortem-cli`` but only shipped an MCP
entrypoint. ``mini-antemortem-cli check ...`` now runs the same
analytical_preflight against on-disk JSONL/JSON inputs and emits text
or JSON.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from omegaprompt.domain.dataset import Dataset, DatasetItem
from omegaprompt.domain.judge import Dimension, HardGate, JudgeRubric
from omegaprompt.domain.params import PromptVariants

from mini_antemortem_cli.cli import main


def _write_dataset(path: Path, n: int, with_ref: bool = False) -> None:
    items = [
        DatasetItem(
            id=f"t{i}",
            input=f"task {i}",
            reference=f"ref {i}" if with_ref else None,
        )
        for i in range(n)
    ]
    ds = Dataset(items=items)
    # Dataset.from_jsonl reads one DatasetItem per line.
    path.write_text(
        "\n".join(item.model_dump_json() for item in ds.items),
        encoding="utf-8",
    )


def _write_rubric(path: Path) -> None:
    rubric = JudgeRubric(
        dimensions=[
            Dimension(name="accuracy", description="is correct", weight=0.7),
            Dimension(name="clarity", description="is clear", weight=0.3),
        ],
        hard_gates=[
            HardGate(name="no_refusal", description="must try", evaluator="judge"),
        ],
    )
    path.write_text(rubric.model_dump_json(), encoding="utf-8")


def _write_variants(path: Path) -> None:
    variants = PromptVariants(
        system_prompts=[
            "You are a precise assistant.",
            "You are a concise assistant. Reply briefly and accurately.",
            "You are a careful assistant who double-checks answers before replying with the final result.",
        ],
        few_shot_examples=[{"input": "1+1", "output": "2"}],
    )
    path.write_text(variants.model_dump_json(), encoding="utf-8")


def _build_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    rubric = tmp_path / "rubric.json"
    variants = tmp_path / "variants.json"
    _write_dataset(train, n=15, with_ref=True)
    _write_dataset(test, n=15, with_ref=True)
    _write_rubric(rubric)
    _write_variants(variants)
    return train, test, rubric, variants


# ---------------------------------------------------------------------------
# Programmatic main() invocation — covers all branches without subprocess.
# ---------------------------------------------------------------------------


def test_cli_check_text_output_succeeds(tmp_path: Path, capsys):
    train, test, rubric, variants = _build_inputs(tmp_path)
    rc = main(
        [
            "check",
            "--target-provider", "openai",
            "--target-model", "gpt-4o",
            "--judge-provider", "anthropic",
            "--judge-model", "claude-opus-4-7",
            "--train", str(train),
            "--test", str(test),
            "--rubric", str(rubric),
            "--variants", str(variants),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    # Each of the seven traps must appear in the human-readable output.
    for trap_id in (
        "self_agreement_bias",
        "small_sample_kc4_power",
        "variants_homogeneous",
        "rubric_weight_concentration",
        "judge_budget_too_small",
        "empty_reference_with_strict_rubric",
        "no_held_out_slice",
    ):
        assert trap_id in captured.out


def test_cli_check_json_output_machine_readable(tmp_path: Path, capsys):
    train, test, rubric, variants = _build_inputs(tmp_path)
    rc = main(
        [
            "check",
            "--target-provider", "openai",
            "--judge-provider", "anthropic",
            "--train", str(train),
            "--test", str(test),
            "--rubric", str(rubric),
            "--variants", str(variants),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "findings" in payload
    assert len(payload["findings"]) == 8  # 7 original + train_test_id_overlap
    for f in payload["findings"]:
        assert {"trap_id", "label", "hypothesis", "severity"}.issubset(f)
    # Reviewer P2: summary fields are part of the JSON envelope.
    assert payload["status"] in {"PASS", "ADVISORY", "HOLD", "BLOCK", "NEEDS_MORE_EVIDENCE"}
    assert payload["highest_severity"] in {"low", "medium", "high", "blocker"}
    assert {"REAL", "GHOST", "NEW", "UNRESOLVED"}.issubset(payload["counts"])


def test_cli_check_no_test_slice_flags_no_held_out(tmp_path: Path, capsys):
    train, _, rubric, variants = _build_inputs(tmp_path)
    rc = main(
        [
            "check",
            "--target-provider", "openai",
            "--judge-provider", "anthropic",
            "--train", str(train),
            "--rubric", str(rubric),
            "--variants", str(variants),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    finding = next(f for f in payload["findings"] if f["trap_id"] == "no_held_out_slice")
    assert finding["label"] == "REAL"
    assert finding["severity"] == "high"


def test_cli_list_traps(capsys):
    rc = main(["list-traps"])
    assert rc == 0
    out = capsys.readouterr().out
    # All 7 traps + their hypotheses appear.
    assert "self_agreement_bias" in out
    assert "no_held_out_slice" in out


def test_cli_version_flag(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "mini-antemortem-cli" in capsys.readouterr().out


def test_cli_no_command_returns_help_with_error_code(capsys):
    """argparse exits with code 2 when the required subcommand is missing."""
    import pytest

    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_cli_fail_on_blocker_currently_returns_zero_when_no_blocker(tmp_path: Path):
    """analytical_preflight never emits BLOCKER today; fail_on_blocker is a
    hook for future trap patterns. Verify the flag does not break the
    happy path."""
    train, test, rubric, variants = _build_inputs(tmp_path)
    rc = main(
        [
            "check",
            "--target-provider", "openai",
            "--judge-provider", "anthropic",
            "--train", str(train),
            "--test", str(test),
            "--rubric", str(rubric),
            "--variants", str(variants),
            "--fail-on-blocker",
        ]
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# Entry-point smoke test — confirms `mini-antemortem-cli` script is wired.
# ---------------------------------------------------------------------------


def test_console_script_resolves():
    """If the user runs `mini-antemortem-cli --version`, it must work.
    We invoke the module entrypoint via -m to avoid PATH dependency."""
    result = subprocess.run(
        [sys.executable, "-m", "mini_antemortem_cli.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "mini-antemortem-cli" in result.stdout
