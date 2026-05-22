from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import publish_readiness


ROOT = Path(__file__).resolve().parents[1]
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"


def _workflow_text() -> str:
    return PUBLISH_WORKFLOW.read_text(encoding="utf-8")


def test_publish_workflow_is_manual_only():
    text = _workflow_text()
    assert "workflow_dispatch:" in text
    assert "on:\n  push:" not in text
    assert "branches:" not in text
    assert "pull_request:" not in text
    assert "release_tag" in text
    assert "v0.4.0" in text
    assert "^v[0-9]+\\.[0-9]+\\.[0-9]+$" in text


def test_publish_workflow_uses_trusted_publishing_oidc_only_on_publish_job():
    text = _workflow_text()
    assert "permissions:\n  contents: read" in text
    publish_section = text.split("\n  publish:\n", 1)[1]
    verify_section = text.split("\n  publish:\n", 1)[0]
    assert "id-token: write" in publish_section
    assert "id-token: write" not in verify_section
    assert "environment: pypi" in publish_section
    assert "pypa/gh-action-pypi-publish@release/v1" in publish_section


def test_publish_workflow_does_not_use_pypi_password_or_token_secret():
    text = _workflow_text().lower()
    forbidden = (
        "password:",
        "__token__",
        "api-token",
        "pypi_token",
        "pypi-token",
        "twine_password",
        "secrets.",
    )
    for marker in forbidden:
        assert marker not in text


def test_publish_workflow_runs_full_release_gate_before_publish():
    text = _workflow_text()
    publish_index = text.index("pypa/gh-action-pypi-publish@release/v1")
    for command in (
        "python -m pytest -q",
        "python scripts/generate_readme_claims.py --check",
        "python scripts/check_repo_consistency.py",
        "python examples/demo_replay.py",
        "python scripts/run_golden_cases.py --check",
        "python scripts/verify_fixture_integrity.py",
        "python scripts/release_audit.py --no-network",
        "python -m build",
        "python scripts/wheel_smoke_install.py dist/*.whl",
        "python scripts/publish_readiness.py --no-network",
    ):
        assert command in text
        assert text.index(command) < publish_index


def test_publish_readiness_fails_closed_when_build_frontend_missing(monkeypatch, capsys):
    def fake_commands() -> list[list[str]]:
        return [[sys.executable, "-m", "build"]]

    def fake_run(cmd, cwd, capture_output, text):
        return subprocess.CompletedProcess(
            cmd,
            returncode=1,
            stdout="",
            stderr="No module named build",
        )

    monkeypatch.setattr(publish_readiness, "readiness_commands", fake_commands)
    monkeypatch.setattr(publish_readiness.subprocess, "run", fake_run)

    assert publish_readiness.main(["--no-network"]) == 1
    err = capsys.readouterr().err
    assert "TOOLING_MISSING" in err
    assert "Python build frontend is unavailable" in err
    assert "PUBLISH_READINESS_OK" not in err
