# mini-antemortem-cli

[English](README.md) · [한국어](README.ko.md) · [English easy](EASY_README.md) · [한국어 easy](EASY_README_KR.md)

calibration 설정의 누출, judge 편향, 유사한 variant 등 11개 구조적 함정을 검사하는 결정적 linter입니다. 로컬 설정에서 `AnalyticalFinding`을 만들며 provider 호출이나 네트워크 접근을 하지 않습니다.

[![CI](https://github.com/hibou04-ops/mini-antemortem-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/hibou04-ops/mini-antemortem-cli/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mini-antemortem-cli.svg)](https://pypi.org/project/mini-antemortem-cli/)

Repository: `hibou04-ops/mini-antemortem-cli` · PyPI: `mini-antemortem-cli` · import: `mini_antemortem_cli` · CLI: `mini-antemortem-cli` · MCP: `mini-antemortem-cli-mcp` with `mini-antemortem-cli[mcp]`

## 0.11.3 — 선택 bridge 계약 회귀 검증

기본 필수 의존성은 `pydantic>=2.6.0` 하나입니다. 필요한 타입·예외·계약은 내부 `_vendor` namespace에 포함하므로 `omegaprompt`, `omega-lock`, NumPy, provider SDK를 설치하지 않아도 import와 CLI가 동작합니다. `omegaprompt>=1.1.0`은 선택 extra `[bridge]`에만 있습니다. 내부 모델과 parent 모델은 별개의 Python 클래스입니다.

Python 3.11+가 필요합니다. 이 checkout의 소스 버전은 **0.11.3**입니다. 아래 명령은 checkout을 설치합니다. GitHub 소스 릴리즈용 변경이며 PyPI 발행은 별도 수동 workflow입니다. 새 버전의 PyPI 게시를 주장하지 않습니다.

```bash
python -m pip install .
mini-antemortem-cli --version
mini-antemortem-cli list-traps --json
python examples/demo_replay.py
```

registry 명령과 fixture 데모는 API 키 없이 실행됩니다. 설치 시 의존성을 내려받을 수 있지만 검사 자체는 오프라인입니다. CLI `check`는 advisory 모드에서 심각한 finding이 있어도 0을 반환할 수 있습니다. CI 정책에는 `--fail-on-severity high`를 명시하세요: 해당 정책 위반은 1, 잘못된 설정은 2입니다.

## 함정 registry

소스: `src/mini_antemortem_cli/traps.py`, `analytical_traps()`. 기존 ID와 안정 코드 `OMG001`~`OMG011`을 유지합니다. `get_canonical_trap_code()`는 알 수 없는 이름에 `KeyError`를 발생시킵니다.

| Trap ID | 검사 내용 |
|---|---|
| `self_agreement_bias` | target과 judge의 vendor family 또는 모델이 같습니다. |
| `small_sample_kc4_power` | hold-out 표본 수가 설정된 임계값보다 작습니다. |
| `variants_homogeneous` | prompt variant의 token 집합 유사도가 높습니다. |
| `rubric_weight_concentration` | 한 rubric 차원의 가중치가 전체를 지배합니다. |
| `judge_budget_too_small` | rubric에 비해 judge 출력 예산이 작습니다. |
| `empty_reference_with_strict_rubric` | reference가 필요한 rubric에 reference가 빠져 있습니다. |
| `no_held_out_slice` | test slice가 제공되지 않았습니다. |
| `train_test_id_overlap` | train/test ID가 겹치거나 ID가 중복됩니다. |
| `routed_provider_opaque_family` | routed provider의 실제 모델 family를 알 수 없습니다. |
| `few_shot_leakage_into_test` | few-shot 입력이 평가 입력과 겹칩니다. 후보 pool 노출과 선택된 입력 노출을 구분합니다. |
| `rubric_dead_weight_dimension` | 가중치가 0인 차원도 judge에 전달됩니다. |

finding 상태는 `REAL`, `GHOST`, `NEW`, `UNRESOLVED`이며 심각도는 `blocker`, `high`, `medium`, `low`입니다. 정상 registry나 lint 결과는 hold-out 일반화·모델 품질·배포 승인을 입증하지 않습니다.

## 입력·정책 계약

few-shot 누출은 Unicode NFC·공백을 정규화한 입력 중복을 검사하고 대소문자를 보존합니다. 서로 다른 입력의 공통 출력 라벨만으로 누출을 판정하지 않습니다. `selected_few_shot_indices`/`--selected-few-shot-indices`가 없으면 후보 pool의 잠재 노출을, 있으면 선택된 입력의 노출을 검사합니다.

`validation_mode`는 `auto`, `paired`, `disjoint`입니다. `paired`에서 ID 집합·입력·reference가 모두 일치할 때 반복측정을 `UNRESOLVED/MEDIUM`으로 구분합니다. 부분 중복이나 충돌은 계속 `REAL/BLOCKER`입니다. `TrapPolicy.fail_on_missing_test=False`는 누락-test finding 정책을 바꾸며 hold-out 측정을 대신하지 않습니다. 알 수 없는 정책 키·잘못된 타입·NaN/Inf·범위 밖 threshold는 입력 오류입니다.

## CLI / Python / MCP

```bash
mini-antemortem-cli check \
  --target-provider openai --target-model gpt-4o \
  --judge-provider anthropic --judge-model claude-opus-4-7 \
  --train examples/demo_config/train.jsonl \
  --test examples/demo_config/test.jsonl \
  --rubric examples/demo_config/rubric.json \
  --variants examples/demo_config/variants.json \
  --judge-output-budget small --fail-on-severity high --json
```

```python
from mini_antemortem_cli import analytical_preflight, analytical_traps

print([trap.id for trap in analytical_traps()])
```

`analytical_preflight(...)`는 로컬 finding 모델을 반환합니다. parent 연동 시 `pip install 'mini-antemortem-cli[bridge]'` 후 `mini_antemortem_cli.bridge.to_omegaprompt_findings(findings)`로 명시적으로 변환하고 parent `PreflightReport`에 넣습니다. 변환은 설치된 parent 모델로 JSON 필드를 검증합니다. 필수 의존성을 되살리는 eager parent import는 없습니다.

```bash
python -m pip install ".[mcp]"
mini-antemortem-cli-mcp
```

MCP 도구는 `analytical_preflight`, `list_traps`입니다. MCP SDK 지원 범위는 `>=1.0.0,<2.0.0`이며 2.x 동작을 보증하지 않습니다. 경로 입력은 `MINI_ANTEMORTEM_WORKSPACE_ROOT` 또는 현재 작업 디렉터리로 제한됩니다. inline JSON은 파일을 읽지 않습니다.

## 로컬 검증

```bash
python -m pytest
python scripts/generate_readme_claims.py --check
python scripts/check_repo_consistency.py
python scripts/run_golden_cases.py --check
python scripts/run_false_positive_audit.py --check
python scripts/verify_fixture_integrity.py
```

이 검증은 로컬 계약과 fixture 회귀를 확인합니다. 외부 실전 데이터셋(OOS)의 효과나 모든 설정에 대한 오탐률을 확립하지 않습니다. vendored 코드의 출처와 라이선스는 [NOTICE](NOTICE) 및 `src/mini_antemortem_cli/_vendor/`에 보존합니다.

## 선택 bridge 계약 검증

실제 설치된 `omegaprompt` 모델과 vendored 모델의 필드·검증 schema·Enum 값을 비교하고 JSON을 양방향으로 왕복합니다. 검증 오류의 종류와 위치도 비교합니다. 아래 검사는 live provider를 호출하지 않습니다.

```bash
python -m pip install ".[dev,mcp,bridge]"
python -m pytest -m bridge tests/test_bridge_contracts.py
```

parent 패키지가 없을 때만 이 선택 통합 검사 파일을 건너뜁니다. 설치된 parent의 import나 계약이 깨졌으면 실패합니다. 독립 core 검사는 parent 없이 계속 실행하며 필수 의존성은 늘리지 않았습니다.

## 근거와 문서

- Generated claims: [English](docs/generated/claims.md) / [한국어](docs/generated/claims_kr.md)
- Trust model: [English](docs/trust_model.md) / [한국어](docs/trust_model_kr.md)
- Toolkit positioning: [English](docs/toolkit_positioning.md) / [한국어](docs/toolkit_positioning_kr.md)
- Claim ledger: [English](docs/claim_ledger.md) / [한국어](docs/claim_ledger_kr.md)
- Examples: [English](docs/examples.md) / [한국어](docs/examples_kr.md)
- [CLI exit codes](docs/cli_exit_codes.md) · [Release checklist](docs/release_checklist.md) · [Changelog](CHANGELOG.md)

한국어 canonical 문서는 [README.ko.md](README.ko.md)이며 [README_KR.md](README_KR.md)는 같은 본문의 호환 사본입니다.

Apache 2.0. [LICENSE](LICENSE) · [NOTICE](NOTICE).
