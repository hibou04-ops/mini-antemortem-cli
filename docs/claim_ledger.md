# Public Claim Ledger

Status markers: `generated`, `source-backed`, `command-backed`,
`artifact-backed`, `qualitative`, `not-claimed`.

| Public claim | Location | Source of truth | Verification command | Status |
|---|---|---|---|---|
| Repository is `hibou04-ops/mini-antemortem-cli`. | README, generated claims | `pyproject.toml` project URLs | `python scripts/generate_readme_claims.py --check` | generated |
| PyPI distribution is `mini-antemortem-cli`. | README, generated claims | `pyproject.toml` `[project].name` | `python scripts/check_repo_consistency.py` | generated |
| Import package is `mini_antemortem_cli`. | README, generated claims | Hatch wheel package path | `python scripts/generate_readme_claims.py --check` | generated |
| Package version is generated from local metadata. | generated claims | `pyproject.toml` and `__init__.__version__` | `python scripts/check_repo_consistency.py` | generated |
| CLI command is `mini-antemortem-cli`. | README, generated claims | `pyproject.toml` `[project.scripts]` | `python scripts/check_repo_consistency.py` | generated |
| CLI subcommands are `check` and `list-traps`. | generated claims, CLI docs | `src/mini_antemortem_cli/cli.py` AST | `python scripts/check_repo_consistency.py` | generated |
| MCP extra is `[mcp]` and command is `mini-antemortem-cli-mcp`. | README, generated claims | `pyproject.toml` optional extras/scripts | `python scripts/check_repo_consistency.py` | generated |
| Built-in trap count and IDs are source-backed. | README, generated claims | `CALIBRATION_TRAPS` / `analytical_traps()` | `python scripts/generate_readme_claims.py --check` | generated |
| Default checks are deterministic and no-network. | README, trust model | Source implementation and no provider-call tests | `python scripts/release_audit.py --no-network` | command-backed |
| Demo replay is deterministic. | README, examples docs | `examples/_demo_output.txt` | `python examples/demo_replay.py` | artifact-backed |
| Golden cases cover the trap registry. | README, generated claims | `benchmarks/golden_cases/expected_cases.json` | `python scripts/run_golden_cases.py --check` | artifact-backed |
| Fixture integrity is checked by SHA-256. | README, release docs | `benchmarks/golden_cases/manifest.sha256` | `python scripts/verify_fixture_integrity.py` | artifact-backed |
| Disk-verified file:line citations are supported. | README | Not implemented in this package | None | not-claimed |
| Provider/model superiority is proven. | README | Not implemented | None | not-claimed |
