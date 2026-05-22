"""Deterministic no-network demo replay for mini-antemortem-cli."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mini_antemortem_cli.cli import main as cli_main  # noqa: E402


CONFIG = ROOT / "examples" / "demo_config"
EXPECTED_PATH = ROOT / "examples" / "_demo_output.txt"

BASE_ARGS = [
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
    str(CONFIG / "train.jsonl"),
    "--test",
    str(CONFIG / "test.jsonl"),
    "--rubric",
    str(CONFIG / "rubric.json"),
    "--variants",
    str(CONFIG / "variants.json"),
    "--judge-output-budget",
    "small",
]


def _rel(path_text: str) -> str:
    path = Path(path_text)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path_text


def _display_command(args: list[str]) -> str:
    display = ["mini-antemortem-cli"]
    for arg in args:
        display.append(_rel(arg) if arg.endswith((".json", ".jsonl")) else arg)
    return " ".join(display)


def _run_cli(args: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli_main(args)
    return rc, buf.getvalue().strip()


def _canonical_json_subset(raw_json: str) -> str:
    payload = json.loads(raw_json)
    findings = [
        {
            "trap_id": finding["trap_id"],
            "label": finding["label"],
            "severity": finding["severity"],
        }
        for finding in payload["findings"]
    ]
    labels = {finding["label"] for finding in findings}
    if "REAL" not in labels or "GHOST" not in labels:
        raise AssertionError("demo must include at least one REAL and one GHOST finding")
    subset = {
        "status": payload["status"],
        "highest_severity": payload["highest_severity"],
        "counts": payload["counts"],
        "findings": findings,
    }
    return json.dumps(subset, indent=2, ensure_ascii=False)


def build_demo_output() -> str:
    text_rc, text_out = _run_cli(BASE_ARGS)
    json_args = [*BASE_ARGS, "--json"]
    json_rc, json_out = _run_cli(json_args)
    canonical_json = _canonical_json_subset(json_out)

    return (
        "# mini-antemortem-cli deterministic demo replay\n\n"
        "Inputs:\n"
        "- train: examples/demo_config/train.jsonl\n"
        "- test: examples/demo_config/test.jsonl\n"
        "- rubric: examples/demo_config/rubric.json\n"
        "- variants: examples/demo_config/variants.json\n\n"
        f"$ {_display_command(BASE_ARGS)}\n"
        f"exit_code: {text_rc}\n"
        f"{text_out}\n\n"
        f"$ {_display_command(json_args)}\n"
        f"exit_code: {json_rc}\n"
        f"{canonical_json}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Rewrite the expected demo fixture.")
    args = parser.parse_args(argv)

    output = build_demo_output()
    if args.update:
        EXPECTED_PATH.write_text(output, encoding="utf-8")
        print(f"updated {EXPECTED_PATH.relative_to(ROOT).as_posix()}")
        return 0

    expected = EXPECTED_PATH.read_text(encoding="utf-8")
    if output != expected:
        print("DEMO_REPLAY_MISMATCH: run python examples/demo_replay.py --update", file=sys.stderr)
        return 1
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

