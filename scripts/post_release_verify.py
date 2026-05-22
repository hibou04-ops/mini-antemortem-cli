"""Post-release verification helper.

Default mode is no-network and does not claim remote release success.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> bool:
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"COMMAND_FAILED: {' '.join(cmd)}", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False
    print(f"OK: {' '.join(cmd)}")
    return True


def _local_checks() -> bool:
    py = sys.executable
    commands = [
        [py, "scripts/generate_readme_claims.py", "--check"],
        [py, "scripts/check_repo_consistency.py"],
        [py, "scripts/verify_fixture_integrity.py"],
    ]
    return all(_run(cmd) for cmd in commands)


def _network_check() -> bool:
    url = "https://pypi.org/pypi/mini-antemortem-cli/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except OSError as exc:
        print(f"ENVIRONMENT_BLOCKED: network check failed: {exc}", file=sys.stderr)
        return False
    version = payload.get("info", {}).get("version")
    if not version:
        print("COMMAND_FAILED: PyPI response missing version", file=sys.stderr)
        return False
    print(f"REMOTE_PYPI_VERSION {version}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", action="store_true", help="Actually query remote PyPI metadata.")
    args = parser.parse_args(argv)

    if not _local_checks():
        return 1
    if args.network:
        return 0 if _network_check() else 1
    print("REMOTE_CHECK_SKIPPED_NO_NETWORK")
    print("POST_RELEASE_LOCAL_VERIFY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

