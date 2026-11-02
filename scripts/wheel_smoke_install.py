"""Install a built wheel into a temporary venv and smoke-test CLI/API."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.repo_facts import project_facts  # noqa: E402


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


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONUTF8"] = "1"
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", env=environment)


def _fail(message: str, result: subprocess.CompletedProcess[str] | None = None) -> int:
    print(message, file=sys.stderr)
    if result is not None:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    return 1


def smoke_wheel(wheel: Path) -> int:
    wheel = wheel.resolve()
    if not wheel.exists():
        return _fail(f"TOOLING_MISSING: wheel not found: {wheel}")

    temp_root = Path(tempfile.mkdtemp(prefix="mini-antemortem-wheel-"))
    try:
        venv = temp_root / "venv"
        result = _run([sys.executable, "-m", "venv", str(venv)], cwd=temp_root)
        if result.returncode != 0:
            return _fail("TOOLING_MISSING: failed to create venv", result)

        py = _venv_python(venv)
        result = _run([str(py), "-m", "pip", "install", "--disable-pip-version-check", str(wheel)], cwd=temp_root)
        if result.returncode != 0:
            return _fail("TOOLING_MISSING: failed to install wheel with pip", result)

        code = (
            "from pathlib import Path; import importlib.util, sys; "
            "import mini_antemortem_cli as m; "
            "assert Path(m.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); "
            "assert all(importlib.util.find_spec(n) is None for n in "
            "('omegaprompt', 'omega_lock', 'numpy', 'openai', 'anthropic')); "
            "print('installed wheel:', m.__file__); print('version:', m.__version__)"
        )
        result = _run([str(py), "-c", code], cwd=temp_root)
        if result.returncode != 0:
            return _fail("TOOLING_MISSING: import smoke failed", result)

        cli = _venv_script(venv, "mini-antemortem-cli")
        if not cli.exists():
            return _fail(f"TOOLING_MISSING: console script missing: {cli}")

        for args in ([str(cli), "--version"], [str(cli), "list-traps"]):
            result = _run(list(args), cwd=temp_root)
            if result.returncode != 0:
                return _fail(f"COMMAND_FAILED: {' '.join(args)}", result)

        cp949_environment = os.environ.copy()
        cp949_environment.pop("PYTHONPATH", None)
        cp949_environment.update(PYTHONUTF8="0", PYTHONIOENCODING="cp949")
        for help_args in (["--help"], ["check", "--help"]):
            help_result = subprocess.run(
                [str(cli), *help_args], cwd=temp_root, capture_output=True,
                env=cp949_environment,
            )
            if help_result.returncode != 0:
                return _fail("COMMAND_FAILED: installed console help failed under cp949")
            help_result.stdout.decode("ascii")

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
            ], cwd=temp_root,
        )
        if result.returncode != 0:
            return _fail("COMMAND_FAILED: CLI JSON check failed", result)
        payload = json.loads(result.stdout)
        expected_count = project_facts(ROOT).trap_count
        actual_count = len(payload.get("findings", []))
        if actual_count != expected_count:
            return _fail(
                f"COMMAND_FAILED: CLI JSON returned {actual_count} findings, "
                f"expected {expected_count} (source trap count)",
                result,
            )

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

