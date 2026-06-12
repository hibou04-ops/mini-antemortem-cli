# mini-antemortem-cli

프롬프트 평가 셋업에 숨은 함정(train/test 누수, judge 편향, 동질적 variant 등)이 가짜 합격 점수를 만들기 전에 잡아내는 결정론적 linter입니다. `omegaprompt` calibration config 입력을 읽어 11가지 source-backed built-in trap 패턴을 분류하고 `AnalyticalFinding` 레코드를 발행합니다. provider 호출도, 네트워크도 사용하지 않습니다. `omegaprompt`와 함께 동작하며, 단독 config linter로도 유용합니다.

[![CI](https://github.com/hibou04-ops/mini-antemortem-cli/actions/workflows/ci.yml/badge.svg?cacheSeconds=3600)](https://github.com/hibou04-ops/mini-antemortem-cli/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mini-antemortem-cli.svg?cacheSeconds=3600)](https://pypi.org/project/mini-antemortem-cli/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg?cacheSeconds=3600)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg?cacheSeconds=3600)](https://www.python.org)

```bash
pip install mini-antemortem-cli
```

Repository: `hibou04-ops/mini-antemortem-cli` · PyPI: `mini-antemortem-cli` · import: `mini_antemortem_cli` · CLI: `mini-antemortem-cli` · MCP: `mini-antemortem-cli-mcp` (`mini-antemortem-cli[mcp]`)

## 0.10.0의 새로운 점

- **새 trap 규칙 2개 — 이제 count는 11개(이전 9개).** `few_shot_leakage_into_test`는 모든 prompt variant에 들어가는 few-shot example의 `input`/`output` 텍스트가 held-out test 또는 train 항목과 일치하는 경우를 flag합니다 — 모델이 추론 시점에 답을 건네받은 셈이라 점수가 일반화가 아닌 암기를 반영합니다(test 겹침 시 `REAL`/`HIGH`, train만 겹치면 `REAL`/`MEDIUM`). `rubric_dead_weight_dimension`은 다른 dimension은 weight가 있는데 어떤 dimension의 weight가 0인 경우를 flag합니다 — 그 죽은 axis는 여전히 judge에게 전송되어(토큰과 주의를 소모) fitness에는 전혀 기여하지 않습니다(`REAL`/`MEDIUM`). 둘 다 결정론적이며 실제 `omegaprompt` 도메인 객체로 도달 가능합니다. false-positive corpus는 53건(0/53)으로 늘었고, 두 새 trap 모두 golden case로 커버됩니다.
- **claim-drift 정정 (수정).** 이 패키지는 trap count를 일관되지 않게 설명했습니다 — `pyproject`는 *nine*, GitHub repository description은 *seven*. source of truth는 `analytical_traps()`이며 이제 11을 반환합니다. 모든 참조(`pyproject`, 4개 README, `__init__` docstring, MCP 서버 instructions, 생성된 claim 문서)가 이 단일 source에서 재생성되고, `scripts/check_repo_consistency.py`가 향후 drift 발생 시 빌드를 실패시킵니다. GitHub repository description도 일치하도록 정정했습니다.
- **버전 무관 publish 워크플로 (수정).** `.github/workflows/publish.yml`이 이제 `pyproject.toml`의 버전을 (`tomllib`로) 읽어 release tag와 `__init__.__version__`이 그것과 일치하는지 빌드 전에 검증합니다. tag/메타데이터 불일치 시 잘못된 버전을 조용히 publish하는 대신 즉시 실패합니다.

## 0.9.1의 새로운 점

- **설정 로드 오류 한 줄 보장 (수정).** bad-schema `--train` / `--test` 행은 `Dataset.from_jsonl`가 다줄 pydantic `ValueError`로 re-raise해서 stderr가 여러 줄로 나왔습니다 — 문서는 한 줄을 약속했습니다. 이제 첫 줄만 출력합니다. 종료 코드는 그대로(`2`)이며 회귀 테스트로 단일 줄 동작을 잠갔습니다.

## 0.9.0의 새로운 점

- **텍스트 모드 verdict 한 줄 (C1):** `check`의 기본(텍스트) 출력이 이제 grep 친화적인 `Summary:` 한 줄로 시작합니다. 기존 native 5단계 status(PASS / ADVISORY / HOLD / BLOCK / NEEDS_MORE_EVIDENCE)를 노출하므로, 지금까지 `--json` 소비자에게만 보이던 핵심 판정을 기본 사용자도 볼 수 있습니다. `... check | head -1`이 곧 CI 신호가 됩니다.
- **Config-referenced citation (C2):** high-signal trap의 finding이 발화한 계산값을 `cite` 필드에 담습니다(겹치는 id, 지배적 rubric dimension과 weight, variant 최대 pairwise Jaccard, 너무 작은 test slice). 디스크 소스 파일이 아니라 제공된 calibration config를 가리킵니다.
- **깔끔한 입력 오류 처리 (C3):** 입력 파일 로드 실패(없는 파일, 잘못된 JSON, 스키마 불일치)는 이제 raw traceback 대신 파일명과 오류 유형을 담은 한 줄 stderr 메시지로 보고하고 종료 코드 `2`(설정 오류, policy gate의 `1`과 구분)로 종료합니다.
- **`list-traps --json` (H1):** trap 레지스트리를 `{id, hypothesis}` JSON 배열로 출력합니다. 텍스트가 기본값으로 유지됩니다.
- **train/test ID 정확 겹침 = BLOCKER (H2, 동작 변경):** held-out 세트가 실제로 held-out이 아닌 hard leak이므로, 정확한 train/test ID 겹침은 이제 `BLOCKER` severity로 발화합니다(이전 `high`). slice 내부 중복은 `medium`으로 유지됩니다. 공개 severity 계약 변경입니다. `--fail-on-severity high`는 여전히 BLOCKER를 잡으므로 CI gate는 회귀하지 않습니다.

이 버전부터 `Development Status :: 4 - Beta`입니다. 이후로 CLI / JSON / MCP surface는 additive-only(추가만, 제거/변경 없음)로 유지하기로 약속합니다.

## 신뢰성 / 검증 링크

- 생성된 source-of-truth claims: [English](docs/generated/claims.md) / [Korean](docs/generated/claims_kr.md)
- 신뢰 모델: [English](docs/trust_model.md) / [Korean](docs/trust_model_kr.md)
- 툴킷 포지셔닝: [English](docs/toolkit_positioning.md) / [Korean](docs/toolkit_positioning_kr.md)
- Claim ledger: [English](docs/claim_ledger.md) / [Korean](docs/claim_ledger_kr.md)
- 예제와 결정론적 데모: [English](docs/examples.md) / [Korean](docs/examples_kr.md)
- 간략 버전: [English](EASY_README.md) / [Korean](EASY_README_KR.md)
- CLI 종료 코드: [docs/cli_exit_codes.md](docs/cli_exit_codes.md)
- 릴리즈 체크리스트: [docs/release_checklist.md](docs/release_checklist.md)
- 릴리즈 후 검증: [docs/post_release_verification.md](docs/post_release_verification.md)
- 영어 README: [README.md](README.md)

## 이럴 때 씁니다

- `omegaprompt` calibration을 돌리기 전에 config의 구조적 결함을 결정론적으로 한 번 더 점검하고 싶을 때.
- 같은 벤더로 묶인 judge bias, 검정력이 부족한 held-out, train/test 누수, routed provider의 모델 패밀리 불명 같은 위험을 CI에서 자동으로 차단하고 싶을 때.
- `derive_adaptation_plan`이 곧바로 소비할 수 있는 기계 판독 가능한 `AnalyticalFinding` 출력을 받고 싶을 때.

## 검증 루프

```bash
python scripts/generate_readme_claims.py --check
python scripts/check_repo_consistency.py
python examples/demo_replay.py
python scripts/run_golden_cases.py --check
python scripts/run_false_positive_audit.py --check
python scripts/verify_fixture_integrity.py
```

위 명령들은 모두 no-network로 설계되어 있습니다. 외부에 공개한 claim, 생성된 docs, 데모 fixture, 골든 케이스, false-positive corpus, 그리고 아티팩트의 SHA-256이 로컬 source of truth와 일치하는지를 확인합니다.

## False-Positive Audit

`benchmarks/false_positive/benign_cases.json`에는 11개 trap 모두에 대한 레이블링된 53건의 benign 구성이 들어 있습니다. 일반 케이스와 경계값 케이스를 함께 포함하며, 분석 단계에서 절대로 발화되어서는 안 되는 입력들입니다. `scripts/run_false_positive_audit.py`는 동일한 결정론적 분류기로 이 corpus를 재생해 trap별 false-positive 비율을 산출하고, 같은 스크립트가 CI 게이트로 묶여 있어 benign 케이스 하나라도 `REAL` / `NEW` / `UNRESOLVED`로 뒤집히면 빌드가 실패합니다. 0.10.0 기준 측정값은 0/53 (0.00%)입니다. 분류기의 알려진 한계는 매니페스트의 `acknowledged_false_positives` 블록에 명시할 수 있어, 게이트가 회귀와 의도된 동작을 구분합니다.

## 결정론적 데모

```bash
python examples/demo_replay.py
```

`examples/demo_config/`의 JSONL/JSON fixture를 읽어 `mini-antemortem-cli check`를 텍스트와 JSON 모드로 실행하고, 재생 결과를 `examples/_demo_output.txt`와 대조합니다. API 키도 네트워크 호출도 필요 없습니다.

## 기존 도구와의 차이점

| 항목 | `mini-antemortem-cli` | `mini-omega-lock` | `antemortem-cli` | `omegaprompt` 기본 경로 | 즉석 리뷰 프롬프트 |
|---|---|---|---|---|---|
| 핵심 역할 | calibration config에 대한 결정론적 analytical preflight. | 실 provider 또는 mock 동작에 대한 empirical preflight. | 더 넓은 pre-diff 정찰과 구현 위험 CLI. | preflight 결과를 받아 돌리는 calibration 엔진. | config에 대한 사람/LLM의 자유 리뷰. |
| 무네트워크 결정론성 | 기본값. | mock 모드는 결정론적, live 모드는 provider 의존. | provider 정찰 켜면 결정론 보장 안 됨. | 설정된 provider 호출 가능. | 보장 없음. |
| Trap 분류 | built-in calibration trap에 한해서 수행. | 이 정적 trap 레지스트리가 아닌 endpoint/judge 동작을 측정. | 넓은 위험 목록을 추론할 수 있음. | preflight 결과 소비만 함. | 프롬프트에 따라 다름. |
| 명시적 trap ID | 모든 finding에 안정적인 `trap_id` 부여. | 이 trap ID 레지스트리 없음. | 자체 evidence/recon 구조 사용. | 받은 analytical finding은 그대로 보존. | 명시 요청 없으면 부재. |
| Source-backed trap 수 | `analytical_traps()`에서 자동 생성. | 해당 없음. | 해당 없음. | 해당 없음. | 없음. |
| Train/test 분리 규율 | held-out slice 누락과 ID 중복을 둘 다 검사. | empirical 동작을 시험할 수는 있으나 분리 무결성을 대체하지 않음. | 설정되면 source/artifact 조사 가능. | caller가 준 데이터셋 그대로 사용. | 놓치기 쉬움. |
| Routed provider 투명성 | routed provider의 패밀리 불명을 `UNRESOLVED`로 표시. | live call이 허용되면 endpoint 동작을 시험할 수 있음. | 설정되면 외부 evidence 수집 가능. | provider family를 추론하지 않음. | provider 레이블 뒤에 가려지기 쉬움. |
| 동일 벤더 judge bias | 같은 패밀리의 target/judge 짝을 표시. | judge 일관성은 측정하지만 이 정적 config 주장은 하지 않음. | 더 넓은 judge 위험 맥락을 분석 가능. | finding이 들어오면 소비. | 주관적인 경우가 많음. |
| CLI/MCP 가용성 | CLI: `mini-antemortem-cli`; MCP: `[mcp]` 익스트라로 `mini-antemortem-cli-mcp`. | 별도 자매 패키지. | 더 넓은 별도 CLI. | 라이브러리 API. | 사용자가 직접 만들지 않으면 없음. |
| 소스 파일 읽기 | 안 함. calibration 입력 파일만 읽음. | 기본적으로 안 함. | 디스크 기반 정찰과 citation을 위해 읽음. | 기본적으로 source 정찰 없음. | 붙여 넣거나 툴 활성화 시에만. |
| Live empirical probe | 없음. | live 모드에서 수행. | 설정되면 수행. | calibration 중 provider 호출. | 가능하나 기본적으로 재현 불가. |
| 디스크 검증된 file:line citation | 없음 (디스크 citation). fixture 무결성만. 0.9.0부터 finding은 trap이 발화한 계산값(겹치는 id, 지배적 rubric dimension 등)을 `cite` 필드에 config-referenced citation으로 담습니다 — 아래 `cite` 행 참조. | 없음. | 그 도구가 evidence-bound citation을 구현한 범위에서 가능. | 없음. | 없음. |
| Config-referenced citation (`cite` 필드) | 있음. 0.9.0부터, 발화값이 계산되는 high-signal trap(겹치는 id, 지배적 rubric dimension과 weight, variant 최대 pairwise Jaccard, 너무 작은 test slice)에 대해 제공됨. 디스크 소스 파일이 아니라 사용자가 제공한 calibration config를 가리킵니다. | 없음. | 별개 메커니즘(디스크 기반). | 없음. | 없음. |
| 증명하지 않는 것 | provider 품질, 프롬프트 우위, 통계적 유효성, 프로덕션 채택, 외부 검증 어느 것도 증명하지 않음. | analytical trap의 부재를 증명하지 않음. | 이 mini 패키지의 trap 수를 증명하지 않음. | 공급되지 않으면 preflight를 수행하지 않음. | 기계적으로 아무것도 증명하지 않음. |

## Built-In Trap 패턴

Source of truth: `src/mini_antemortem_cli/traps.py`의 `analytical_traps()`.

| Trap ID | 점검 내용 |
|---|---|
| `self_agreement_bias` | target과 judge가 같은 벤더 패밀리거나 동일 모델이라 self-agreement 위험이 있는지. |
| `small_sample_kc4_power` | held-out 표본 수가 KC-4/Pearson 신호의 통계적 검정력을 확보하기에 충분한지. |
| `variants_homogeneous` | 프롬프트 변형들이 의미 있는 sensitivity 신호를 만들 만큼 서로 다른지. |
| `rubric_weight_concentration` | 한 dimension이 weighted fitness를 과도하게 지배하지는 않는지. |
| `judge_budget_too_small` | rubric의 dimension과 gate 수에 비해 SMALL judge output budget이 부족하지는 않은지. |
| `empty_reference_with_strict_rubric` | rubric이 ground truth와의 비교를 암시하는데 데이터셋의 reference가 없는지. |
| `no_held_out_slice` | test slice가 없어 walk-forward 검증 자체가 불가능한지. |
| `train_test_id_overlap` | train/test ID가 겹치거나 중복돼 per-item 상관이 신뢰하기 어려운지. |
| `routed_provider_opaque_family` | routed provider가 실제로 서빙되는 모델의 패밀리를 가려서 검사가 막히는지. |
| `few_shot_leakage_into_test` | prompt에 박힌 few-shot example의 input/output이 held-out 항목과 겹쳐 암기로 점수가 부풀려지는지. |
| `rubric_dead_weight_dimension` | 다른 dimension은 weight가 있는데 weight 0인 dimension이 있어 judge에게는 전송되나 점수에는 반영되지 않는지. |

각 finding은 `REAL`, `GHOST`, `NEW`, `UNRESOLVED` 중 하나의 라벨과 `blocker`, `high`, `medium`, `low` 중 하나의 severity를 가집니다.

## CLI

```bash
mini-antemortem-cli list-traps
mini-antemortem-cli check \
  --target-provider openai \
  --target-model gpt-4o \
  --judge-provider anthropic \
  --judge-model claude-opus-4-7 \
  --train examples/demo_config/train.jsonl \
  --test examples/demo_config/test.jsonl \
  --rubric examples/demo_config/rubric.json \
  --variants examples/demo_config/variants.json \
  --judge-output-budget small
```

기계 판독 가능한 출력은 `--json`으로, CI에서 high 이상의 `REAL`/`UNRESOLVED` finding을 실패로 처리하려면 `--fail-on-severity high`로 받습니다(BLOCKER도 함께 잡힙니다). 하위 호환을 위한 `--fail-on-blocker` 별칭은 남아 있으며, 0.9.0부터 train/test ID 정확 겹침이 BLOCKER로 발화하므로 실제 실패에서 작동합니다.

## Python API

```python
from mini_antemortem_cli import analytical_preflight, analytical_traps
```

`analytical_preflight(...)`의 반환은 `omegaprompt.preflight.contracts.AnalyticalFinding` 객체이고, `omegaprompt.preflight.PreflightReport`와 `derive_adaptation_plan`에 그대로 넘길 수 있습니다.

## MCP

```bash
pip install "mini-antemortem-cli[mcp]"
mini-antemortem-cli-mcp
# 또는
python -m mini_antemortem_cli.mcp
```

MCP 서버는 `analytical_preflight`와 `list_traps`를 노출합니다. 경로 입력은 `MINI_ANTEMORTEM_WORKSPACE_ROOT` 또는 현재 작업 디렉터리로 제한되며, 인라인 JSON 객체는 파일 시스템을 건드리지 않습니다.

## 릴리즈 위생

```bash
python scripts/release_audit.py --no-network
python -m build
python scripts/wheel_smoke_install.py dist/*.whl
python scripts/publish_readiness.py --no-network
```

이 스크립트들은 publish, tag, GitHub release를 만들지 않습니다. 실제 publish는 [.github/workflows/publish.yml](.github/workflows/publish.yml)에서 manual dispatch로만 진행되며, PyPI Trusted Publishing / GitHub OIDC로 토큰 시크릿 없이 동작합니다. 순서와 셋업은 [docs/release_checklist.md](docs/release_checklist.md)에 있습니다.

## 라이선스

Apache 2.0. [LICENSE](LICENSE)를 참고하세요.

License 이력: PyPI 버전 0.1.0은 MIT `LICENSE` 파일로 배포되었습니다. 저장소는 2026-04-22(`d2d7eb7`)에 Apache 2.0으로 재라이선스되었으며, 0.2.0 이후 모든 버전은 Apache 2.0으로 배포됩니다. 0.1.0을 설치한 사용자는 그 사본에 한해 MIT 라이선스를 보유하며, 라이선스 변경은 소급 적용되지 않습니다.
