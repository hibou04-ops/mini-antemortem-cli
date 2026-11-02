"""Exercise the standalone demo script without importing an examples package."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PATH = ROOT / "examples" / "_demo_output.txt"


def _demo_output():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "demo_replay.py")],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_demo_replay_fixture_is_current():
    assert _demo_output() == EXPECTED_PATH.read_text(encoding="utf-8")


def test_demo_replay_contains_real_and_ghost_findings():
    output = _demo_output()
    assert '"label": "REAL"' in output
    assert '"label": "GHOST"' in output
    assert "mini-antemortem-cli check" in output

