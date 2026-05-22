# Public Claim Ledger

Status marker: `generated`, `source-backed`, `command-backed`, `artifact-backed`, `qualitative`, `not-claimed`.

| Public claim | Location | Source of truth | Verification command | Status |
|---|---|---|---|---|
| Repository는 `hibou04-ops/mini-antemortem-cli`. | README, generated claims | `pyproject.toml` project URLs | `python scripts/generate_readme_claims.py --check` | generated |
| PyPI distribution은 `mini-antemortem-cli`. | README, generated claims | `pyproject.toml` `[project].name` | `python scripts/check_repo_consistency.py` | generated |
| Import package는 `mini_antemortem_cli`. | README, generated claims | Hatch wheel package path | `python scripts/generate_readme_claims.py --check` | generated |
| Package version은 local metadata에서 생성. | generated claims | `pyproject.toml`, `__init__.__version__` | `python scripts/check_repo_consistency.py` | generated |
| CLI command는 `mini-antemortem-cli`. | README, generated claims | `pyproject.toml` `[project.scripts]` | `python scripts/check_repo_consistency.py` | generated |
| CLI subcommands는 `check`, `list-traps`. | generated claims, CLI docs | `src/mini_antemortem_cli/cli.py` AST | `python scripts/check_repo_consistency.py` | generated |
| MCP extra는 `[mcp]`, command는 `mini-antemortem-cli-mcp`. | README, generated claims | `pyproject.toml` optional extras/scripts | `python scripts/check_repo_consistency.py` | generated |
| Built-in trap count와 IDs는 source-backed. | README, generated claims | `CALIBRATION_TRAPS` / `analytical_traps()` | `python scripts/generate_readme_claims.py --check` | generated |
| Default check는 deterministic/no-network. | README, trust model | Source implementation과 no provider-call tests | `python scripts/release_audit.py --no-network` | command-backed |
| Demo replay는 deterministic. | README, examples docs | `examples/_demo_output.txt` | `python examples/demo_replay.py` | artifact-backed |
| Golden cases는 trap registry를 cover. | README, generated claims | `benchmarks/golden_cases/expected_cases.json` | `python scripts/run_golden_cases.py --check` | artifact-backed |
| Fixture integrity는 SHA-256으로 check. | README, release docs | `benchmarks/golden_cases/manifest.sha256` | `python scripts/verify_fixture_integrity.py` | artifact-backed |
| Disk-verified file:line citation 지원. | README | 이 package에는 없음 | None | not-claimed |
| Provider/model superiority 증명. | README | 구현하지 않음 | None | not-claimed |
