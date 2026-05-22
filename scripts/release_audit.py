"""No-network release audit for mini-antemortem-cli."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _classify_failure(output: str) -> str:
    tooling_markers = (
        "No module named",
        "ModuleNotFoundError",
        "not recognized",
        "No Python at",
        "pytest: command not found",
    )
    if any(marker in output for marker in tooling_markers):
        return "TOOLING_MISSING"
    return "COMMAND_FAILED"


def run_command(cmd: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError as exc:
        return False, f"TOOLING_MISSING: {cmd[0]}: {exc}"
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        return False, f"{_classify_failure(output)}: {' '.join(cmd)}\n{output}"
    return True, f"OK: {' '.join(cmd)}"


def audit_commands() -> list[list[str]]:
    py = sys.executable
    return [
        [py, "scripts/generate_readme_claims.py", "--check"],
        [py, "scripts/check_repo_consistency.py"],
        [py, "examples/demo_replay.py"],
        [py, "scripts/run_golden_cases.py", "--check"],
        [py, "scripts/verify_fixture_integrity.py"],
        [py, "-m", "pytest", "-q"],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-network", action="store_true", help="Document that audit is offline.")
    parser.parse_args(argv)

    failures: list[str] = []
    for cmd in audit_commands():
        ok, message = run_command(cmd)
        print(message)
        if not ok:
            failures.append(message)
    if failures:
        return 1
    print("RELEASE_AUDIT_OK no-network")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

