"""Static repository facts used by generated docs and release checks.

This module deliberately avoids importing ``mini_antemortem_cli``. Importing
the package requires runtime dependencies such as ``omegaprompt``; consistency
checks should still run in a minimal no-network environment.
"""

from __future__ import annotations

import ast
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

COUNT_WORDS_EN = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


@dataclass(frozen=True)
class TrapFact:
    id: str
    hypothesis: str


@dataclass(frozen=True)
class RepoFacts:
    repository_name: str
    distribution_name: str
    install_command: str
    import_package: str
    version: str
    cli_command: str
    cli_subcommands: tuple[str, ...]
    mcp_extra: str
    mcp_command: str
    mcp_module_command: str
    traps: tuple[TrapFact, ...]

    @property
    def trap_count(self) -> int:
        return len(self.traps)

    @property
    def trap_ids(self) -> tuple[str, ...]:
        return tuple(t.id for t in self.traps)

    @property
    def trap_count_word(self) -> str:
        return COUNT_WORDS_EN.get(self.trap_count, str(self.trap_count))


def read_pyproject(root: Path = ROOT) -> dict:
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def _literal_string(node: ast.AST) -> str:
    value = ast.literal_eval(node)
    if not isinstance(value, str):
        raise TypeError(f"expected string literal, got {type(value).__name__}")
    return value


def parse_trap_patterns(root: Path = ROOT) -> tuple[TrapFact, ...]:
    source = (root / "src" / "mini_antemortem_cli" / "traps.py").read_text(
        encoding="utf-8"
    )
    module = ast.parse(source)
    for stmt in module.body:
        value: ast.AST | None = None
        if isinstance(stmt, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CALIBRATION_TRAPS" for t in stmt.targets
        ):
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "CALIBRATION_TRAPS":
            value = stmt.value
        if value is None:
            continue
        if not isinstance(value, ast.Tuple):
            raise ValueError("CALIBRATION_TRAPS must be a tuple literal")
        traps: list[TrapFact] = []
        for elt in value.elts:
            if not isinstance(elt, ast.Call):
                continue
            kwargs = {kw.arg: kw.value for kw in elt.keywords if kw.arg}
            traps.append(
                TrapFact(
                    id=_literal_string(kwargs["id"]),
                    hypothesis=_literal_string(kwargs["hypothesis"]),
                )
            )
        if not traps:
            raise ValueError("CALIBRATION_TRAPS contained no TrapPattern calls")
        return tuple(traps)
    raise ValueError("CALIBRATION_TRAPS assignment not found")


def parse_cli_subcommands(root: Path = ROOT) -> tuple[str, ...]:
    source = (root / "src" / "mini_antemortem_cli" / "cli.py").read_text(
        encoding="utf-8"
    )
    module = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "add_parser" and node.args:
            arg0 = node.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                names.append(arg0.value)
    return tuple(dict.fromkeys(names))


def parse_init_version(root: Path = ROOT) -> str:
    source = (root / "src" / "mini_antemortem_cli" / "__init__.py").read_text(
        encoding="utf-8"
    )
    match = re.search(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", source, re.M)
    if not match:
        raise ValueError("__version__ assignment not found")
    return match.group(1)


def project_facts(root: Path = ROOT) -> RepoFacts:
    pyproject = read_pyproject(root)
    project = pyproject["project"]
    scripts = project.get("scripts", {})
    optional = project.get("optional-dependencies", {})
    wheel_packages = (
        pyproject.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("packages", [])
    )
    import_package = Path(wheel_packages[0]).name if wheel_packages else "mini_antemortem_cli"
    homepage = project.get("urls", {}).get("Repository") or project.get("urls", {}).get("Homepage", "")
    repository_name = homepage.rstrip("/").removeprefix("https://github.com/")
    distribution_name = project["name"]
    return RepoFacts(
        repository_name=repository_name,
        distribution_name=distribution_name,
        install_command=f"pip install {distribution_name}",
        import_package=import_package,
        version=project["version"],
        cli_command="mini-antemortem-cli",
        cli_subcommands=parse_cli_subcommands(root),
        mcp_extra="mcp" if "mcp" in optional else "",
        mcp_command="mini-antemortem-cli-mcp",
        mcp_module_command=f"python -m {import_package}.mcp",
        traps=parse_trap_patterns(root),
    )


def markdown_links(markdown: str) -> list[str]:
    return re.findall(r"\[[^\]]+\]\(([^)]+)\)", markdown)


def local_link_target(link: str) -> str | None:
    if (
        link.startswith("http://")
        or link.startswith("https://")
        or link.startswith("mailto:")
        or link.startswith("#")
    ):
        return None
    target = link.split("#", 1)[0].strip()
    if not target:
        return None
    return target
