"""No-network publish readiness gate.

This script never publishes, tags, or creates a GitHub release.
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _classify(output: str) -> str:
    if "No module named" in output or "ModuleNotFoundError" in output:
        return "TOOLING_MISSING"
    return "COMMAND_FAILED"


def _tooling_hint(cmd: list[str], output: str) -> str:
    if cmd[-2:] == ["-m", "build"] and "No module named build" in output:
        return (
            "\nTOOLING_MISSING_DETAIL: Python build frontend is unavailable. "
            "Install the dev extra with `python -m pip install -e \".[dev,mcp]\"` "
            "or install `build>=1.2.0`, then rerun publish readiness."
        )
    return ""


def _run(cmd: list[str]) -> bool:
    try:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError as exc:
        print(f"TOOLING_MISSING: {cmd[0]}: {exc}", file=sys.stderr)
        return False
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        print(f"{_classify(output)}: {' '.join(cmd)}", file=sys.stderr)
        print(output, file=sys.stderr)
        hint = _tooling_hint(cmd, output)
        if hint:
            print(hint, file=sys.stderr)
        return False
    print(f"OK: {' '.join(cmd)}")
    return True


def readiness_commands() -> list[list[str]]:
    py = sys.executable
    return [
        [py, "-m", "pytest", "-q"],
        [py, "scripts/generate_readme_claims.py", "--check"],
        [py, "scripts/check_repo_consistency.py"],
        [py, "examples/demo_replay.py"],
        [py, "scripts/run_golden_cases.py", "--check"],
        [py, "scripts/verify_fixture_integrity.py"],
        [py, "scripts/release_audit.py", "--no-network"],
        [py, "-m", "build"],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-network", action="store_true", help="Required release mode.")
    parser.parse_args(argv)

    for cmd in readiness_commands():
        if not _run(cmd):
            return 1

    wheels = sorted(glob.glob(str(ROOT / "dist" / "*.whl")))
    if not wheels:
        print("TOOLING_MISSING: no built wheel in dist/", file=sys.stderr)
        return 1
    if not _run([sys.executable, "scripts/wheel_smoke_install.py", wheels[-1]]):
        return 1

    print("PUBLISH_READINESS_OK no-network")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
