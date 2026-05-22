"""Install a built wheel into a temporary venv and smoke-test CLI/API."""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _expand_wheels(args: list[str]) -> list[Path]:
    wheels: list[Path] = []
    for arg in args:
        matches = glob.glob(arg)
        if matches:
            wheels.extend(Path(match) for match in matches)
        else:
            wheels.append(Path(arg))
    return sorted(wheels)


def _venv_python(venv: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _venv_script(venv: Path, name: str) -> Path:
    if sys.platform.startswith("win"):
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


def _run(cmd: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _fail(message: str, result: subprocess.CompletedProcess[str] | None = None) -> int:
    print(message, file=sys.stderr)
    if result is not None:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    return 1


def smoke_wheel(wheel: Path) -> int:
    if not wheel.exists():
        return _fail(f"TOOLING_MISSING: wheel not found: {wheel}")

    temp_root = Path(tempfile.mkdtemp(prefix="mini-antemortem-wheel-"))
    try:
        venv = temp_root / "venv"
        result = _run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)])
        if result.returncode != 0:
            return _fail("TOOLING_MISSING: failed to create venv", result)

        py = _venv_python(venv)
        result = _run([str(py), "-m", "pip", "install", "--no-deps", str(wheel)])
        if result.returncode != 0:
            return _fail("TOOLING_MISSING: failed to install wheel with pip", result)

        result = _run([str(py), "-c", "import mini_antemortem_cli; print(mini_antemortem_cli.__version__)"])
        if result.returncode != 0:
            return _fail("TOOLING_MISSING: import smoke failed", result)

        cli = _venv_script(venv, "mini-antemortem-cli")
        if not cli.exists():
            return _fail(f"TOOLING_MISSING: console script missing: {cli}")

        for args in ([str(cli), "--version"], [str(cli), "list-traps"]):
            result = _run(list(args))
            if result.returncode != 0:
                return _fail(f"COMMAND_FAILED: {' '.join(args)}", result)

        config = ROOT / "examples" / "demo_config"
        result = _run(
            [
                str(cli),
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
                "--json",
            ]
        )
        if result.returncode != 0:
            return _fail("COMMAND_FAILED: CLI JSON check failed", result)
        payload = json.loads(result.stdout)
        if len(payload.get("findings", [])) != 9:
            return _fail("COMMAND_FAILED: CLI JSON did not return 9 findings", result)

        print(f"WHEEL_SMOKE_OK {wheel}")
        return 0
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+")
    args = parser.parse_args(argv)
    wheels = _expand_wheels(args.wheels)
    if not wheels:
        return _fail("TOOLING_MISSING: no wheels matched")
    return smoke_wheel(wheels[-1])


if __name__ == "__main__":
    raise SystemExit(main())

