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

## 0.11.0 검증된 추가 계약

| 주장 | 근거 | 재현 | 상태 |
|---|---|---|---|
| 개발 사례 7개에서 label/식별자 오탐 4개 교정, 실제 입력 노출 2개 유지 | `benchmarks/upgrade-0.11.0/*.json` | `python benchmarks/precision_cost.py` | artifact-backed |
| 1,000개 같은 prompt의 최대값1 유지, 정규화1회·union/intersection0회 | `benchmarks/precision_cost.py` | `python benchmarks/precision_cost.py` | command-backed |
| 정책 오류·missing test·paired 내용 확인·선택된 입력 노출 | `tests/test_precision_contracts.py` | `python -m pytest -q tests/test_precision_contracts.py` | source-backed |
| 일반 오탐률/실제 모델 품질 개선 | 확립되지 않음 | 해당 없음 | not-claimed |
