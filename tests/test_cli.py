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
    # Each built-in trap must appear in the human-readable output.
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
    assert len(payload["findings"]) == 9
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
    # Built-in traps + their hypotheses appear.
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


def test_cli_fail_on_blocker_trips_on_overlap_blocker(tmp_path: Path):
    """H2 (0.9.0): _build_inputs writes train and test with identical ids
    (t0..t14), so train_test_id_overlap fires at BLOCKER severity. The
    --fail-on-blocker gate (REAL,NEW,UNRESOLVED at blocker) must now trip,
    returning 1 — the orphaned BLOCKER enum is no longer a no-op."""
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
    assert rc == 1


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


# ---------------------------------------------------------------------------
# C1 (0.9.0): text-mode summary/verdict line surfacing the native status.
# ---------------------------------------------------------------------------


def test_cli_text_output_leads_with_summary_status_line(tmp_path: Path, capsys):
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
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    first_line = out.splitlines()[0]
    assert first_line.startswith("Summary:")
    # Native 5-level status must appear on the summary line.
    assert any(
        s in first_line
        for s in ("PASS", "ADVISORY", "HOLD", "BLOCK", "NEEDS_MORE_EVIDENCE")
    )


# ---------------------------------------------------------------------------
# C3 (0.9.0): structured file-load errors -> exit code 2 (config error),
# message names the file as the user gave it. Distinct from policy gate (1).
# ---------------------------------------------------------------------------


def test_cli_missing_input_file_exits_2_and_names_file(tmp_path: Path, capsys):
    _, test, rubric, variants = _build_inputs(tmp_path)
    missing = tmp_path / "does_not_exist.jsonl"
    rc = main(
        [
            "check",
            "--target-provider", "openai",
            "--judge-provider", "anthropic",
            "--train", str(missing),
            "--test", str(test),
            "--rubric", str(rubric),
            "--variants", str(variants),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "does_not_exist.jsonl" in err


def test_cli_malformed_json_exits_2_and_names_file(tmp_path: Path, capsys):
    train, test, _, variants = _build_inputs(tmp_path)
    bad_rubric = tmp_path / "bad_rubric.json"
    bad_rubric.write_text("{not valid json", encoding="utf-8")
    rc = main(
        [
            "check",
            "--target-provider", "openai",
            "--judge-provider", "anthropic",
            "--train", str(train),
            "--test", str(test),
            "--rubric", str(bad_rubric),
            "--variants", str(variants),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "bad_rubric.json" in err


def test_cli_bad_schema_exits_2_and_names_file(tmp_path: Path, capsys):
    train, test, rubric, _ = _build_inputs(tmp_path)
    bad_variants = tmp_path / "bad_variants.json"
    bad_variants.write_text('{"unexpected": 1}', encoding="utf-8")
    rc = main(
        [
            "check",
            "--target-provider", "openai",
            "--judge-provider", "anthropic",
            "--train", str(train),
            "--test", str(test),
            "--rubric", str(rubric),
            "--variants", str(bad_variants),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "bad_variants.json" in err


# ---------------------------------------------------------------------------
# H1 (0.9.0): list-traps --json emits an array; text stays the default.
# ---------------------------------------------------------------------------


def test_cli_list_traps_json_emits_array(capsys):
    rc = main(["list-traps", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert len(payload) == 9
    for entry in payload:
        assert set(entry.keys()) == {"id", "hypothesis"}


def test_cli_list_traps_text_default_unchanged(capsys):
    rc = main(["list-traps"])
    assert rc == 0
    out = capsys.readouterr().out
    # Text default: not JSON, trap ids present line-by-line.
    assert not out.lstrip().startswith("[")
    assert "self_agreement_bias" in out


# ---------------------------------------------------------------------------
# H2 (0.9.0) RUN-VERIFICATION: actually run the CLI as a subprocess against an
# overlap fixture and assert the *process* exit code — confirm --fail-on-
# severity high still trips on BLOCKER (no CI-gate regression).
# ---------------------------------------------------------------------------


def _write_overlap_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    rubric = tmp_path / "rubric.json"
    variants = tmp_path / "variants.json"
    train_ids = ["shared"] + [f"t{i}" for i in range(11)]
    test_ids = ["shared"] + [f"v{i}" for i in range(11)]
    train.write_text(
        "\n".join(
            DatasetItem(id=i, input=f"task {i}", reference=f"ref {i}").model_dump_json()
            for i in train_ids
        ),
        encoding="utf-8",
    )
    test.write_text(
        "\n".join(
            DatasetItem(id=i, input=f"task {i}", reference=f"ref {i}").model_dump_json()
            for i in test_ids
        ),
        encoding="utf-8",
    )
    _write_rubric(rubric)
    _write_variants(variants)
    return train, test, rubric, variants


def test_cli_subprocess_fail_on_severity_high_trips_on_overlap_blocker(tmp_path: Path):
    """ADVISOR-MANDATED run-verification (not reconstruction): launch the real
    CLI process against an overlap fixture and assert the actual process exit
    code is 1 — the gate still fires on BLOCKER (4 >= 3)."""
    train, test, rubric, variants = _write_overlap_inputs(tmp_path)
    result = subprocess.run(
        [
            sys.executable, "-m", "mini_antemortem_cli.cli", "check",
            "--target-provider", "openai",
            "--judge-provider", "anthropic",
            "--train", str(train),
            "--test", str(test),
            "--rubric", str(rubric),
            "--variants", str(variants),
            "--fail-on-severity", "high",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    # The overlap finding must be visible as BLOCKER in the text output.
    assert "BLOCKER" in result.stdout
