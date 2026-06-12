"""Repository consistency checks for public claims."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.generate_readme_claims import build_claim_docs
from scripts.repo_facts import ROOT, local_link_target, markdown_links, parse_init_version, project_facts


REQUIRED_README_LINKS = (
    "docs/generated/claims.md",
    "docs/generated/claims_kr.md",
    "docs/trust_model.md",
    "docs/trust_model_kr.md",
    "docs/toolkit_positioning.md",
    "docs/toolkit_positioning_kr.md",
    "docs/claim_ledger.md",
    "docs/claim_ledger_kr.md",
    "docs/examples.md",
    "docs/examples_kr.md",
    "EASY_README.md",
    "EASY_README_KR.md",
)

PUBLIC_DOCS = (
    "README.md",
    "README_KR.md",
    "EASY_README.md",
    "EASY_README_KR.md",
)


def _read(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8")


def _has_exact_test_count(text: str) -> bool:
    patterns = (
        r"tests[-_ ]*\d+",
        r"\d+\s+passing",
        r"\d+\s+tests?\s+passing",
    )
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def _has_static_pypi_version_badge(text: str) -> bool:
    return bool(re.search(r"img\.shields\.io/badge/pypi-\d+\.\d+\.\d+", text, flags=re.I))


def run_checks(root: Path = ROOT) -> list[str]:
    facts = project_facts(root)
    errors: list[str] = []

    readme = _read(root, "README.md")
    init_text = _read(root, "src/mini_antemortem_cli/__init__.py")
    cli_text = _read(root, "src/mini_antemortem_cli/cli.py")
    mcp_text = _read(root, "src/mini_antemortem_cli/mcp/server.py")

    if parse_init_version(root) != facts.version:
        errors.append("__init__.__version__ does not match pyproject.toml")

    for rel, expected in build_claim_docs(facts).items():
        path = root / rel
        if not path.exists():
            errors.append(f"generated doc missing: {rel}")
        elif path.read_text(encoding="utf-8") != expected:
            errors.append(f"generated doc stale: {rel}")

    for required in REQUIRED_README_LINKS:
        if required not in readme:
            errors.append(f"README missing required link to {required}")

    for link in markdown_links(readme):
        target = local_link_target(link)
        if target is None:
            continue
        target_path = root / target
        if not target_path.exists():
            errors.append(f"README local link target does not exist: {link}")

    if _has_static_pypi_version_badge(readme):
        errors.append("README uses a hard-coded PyPI version badge")
    if _has_exact_test_count(readme):
        errors.append("README contains an unsupported exact test-count claim")

    common_fact_values = (
        facts.repository_name,
        facts.distribution_name,
        facts.import_package,
        facts.cli_command,
        facts.mcp_command,
    )
    for rel in PUBLIC_DOCS:
        text = _read(root, rel)
        for value in common_fact_values:
            if value not in text:
                errors.append(f"{rel} missing public naming fact: {value}")
        if facts.mcp_extra and f"[{facts.mcp_extra}]" not in text:
            errors.append(f"{rel} missing MCP extra marker: [{facts.mcp_extra}]")

    stale_patterns = (
        "seven calibration",
        "seven trap",
        "seven built-in",
        "7 trap",
        "7 known",
        "7개",
    )
    for rel, text in (
        ("README.md", readme),
        ("README_KR.md", _read(root, "README_KR.md")),
        ("EASY_README.md", _read(root, "EASY_README.md")),
        ("EASY_README_KR.md", _read(root, "EASY_README_KR.md")),
        ("src/mini_antemortem_cli/__init__.py", init_text),
        ("src/mini_antemortem_cli/cli.py", cli_text),
        ("src/mini_antemortem_cli/mcp/server.py", mcp_text),
    ):
        lowered = text.lower()
        for pattern in stale_patterns:
            if pattern in lowered:
                errors.append(f"{rel} contains stale trap-count wording: {pattern}")

    count_markers = (str(facts.trap_count), facts.trap_count_word)
    if not any(marker in readme for marker in count_markers):
        errors.append("README does not mention the source trap count")
    if not any(marker in init_text for marker in count_markers):
        errors.append("__init__.py docstring does not mention the source trap count")

    # Docs that enumerate the trap IDs explicitly must stay in lock-step with
    # the source registry. trust_model.* carry a "Current trap IDs" list that
    # the generator does not own, so they used to be able to drift silently —
    # the very failure mode this tooling exists to prevent. Enforce them here.
    for rel in PUBLIC_DOCS + (
        "docs/generated/claims.md",
        "docs/generated/claims_kr.md",
        "docs/trust_model.md",
        "docs/trust_model_kr.md",
    ):
        text = _read(root, rel)
        for trap_id in facts.trap_ids:
            if trap_id not in text:
                errors.append(f"{rel} missing trap id: {trap_id}")

    if tuple(facts.cli_subcommands) != ("check", "list-traps"):
        errors.append(f"CLI subcommands drifted: {facts.cli_subcommands}")
    if facts.cli_command not in _read(root, "pyproject.toml"):
        errors.append("CLI command missing from pyproject scripts")
    if facts.mcp_command not in _read(root, "pyproject.toml"):
        errors.append("MCP command missing from pyproject scripts")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    errors = run_checks()
    if errors:
        for error in errors:
            print(f"REPO_CONSISTENCY_FAIL: {error}", file=sys.stderr)
        return 1
    print("REPO_CONSISTENCY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
