"""Self-contained offline fixtures exercising the installed CLI policy gate.

The printed lines summarize verified CLI JSON and exit codes. They are not
the full ``check`` output and do not measure evaluator or model quality.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from mini_antemortem_cli.cli import main as cli_main
from mini_antemortem_cli.traps import analytical_traps


def _write_dataset(path: Path, prefix: str) -> None:
    items = [
        {"id": f"{prefix}-{i}", "input": f"{prefix} question {i}", "reference": f"answer {i}"}
        for i in range(20)
    ]
    path.write_text("\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8")


def main() -> int:
    with TemporaryDirectory(prefix="mini-antemortem-demo-") as directory:
        root = Path(directory)
        train, test = root / "train.jsonl", root / "test.jsonl"
        rubric, variants = root / "rubric.json", root / "variants.json"
        _write_dataset(train, "train")
        rubric.write_text(json.dumps({"dimensions": [
            {"name": "accuracy", "description": "Matches the reference", "weight": 0.5},
            {"name": "clarity", "description": "Clear explanation", "weight": 0.5},
        ]}), encoding="utf-8")
        variants.write_text(json.dumps({"system_prompts": [
            "Explain arithmetic carefully.",
            "Summarize assumptions before deriving an answer.",
            "Verify boundary cases and return concise results.",
        ], "few_shot_examples": []}), encoding="utf-8")
        args = [
            "check", "--target-provider", "openai", "--judge-provider", "anthropic",
            "--train", str(train), "--test", str(test), "--rubric", str(rubric),
            "--variants", str(variants), "--validation-mode", "disjoint",
            "--judge-output-budget", "large", "--fail-on-severity", "high", "--json",
        ]
        for prefix, expected_exit, expected_label, label, description in [
            ("train", 1, "REAL", "FAIL", "overlapping train/test IDs"),
            ("test", 0, "GHOST", "PASS", "disjoint train/test IDs"),
        ]:
            _write_dataset(test, prefix)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = cli_main(args)
            payload = json.loads(output.getvalue())
            overlap = next(f for f in payload["findings"] if f["trap_id"] == "train_test_id_overlap")
            if exit_code != expected_exit or overlap["label"] != expected_label:
                raise RuntimeError(f"Unexpected CLI result for {description}: {exit_code}, {overlap}")
            if len(payload["findings"]) != len(analytical_traps()):
                raise RuntimeError("CLI result does not cover the complete trap registry")
            print(f"[{label}] {description}: gate exit {exit_code}")
        print(f"[PASS] checked {len(analytical_traps())} local traps per fixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
