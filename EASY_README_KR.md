# mini-antemortem-cli — 빠른 시작

[English](README.md) · [한국어](README.ko.md) · [English easy](EASY_README.md) · [한국어 easy](EASY_README_KR.md)

calibration 설정의 누출, judge 편향, 유사한 variant 등 11개 구조적 함정을 검사하는 결정적 linter입니다. 로컬 설정에서 `AnalyticalFinding`을 만들며 provider 호출이나 네트워크 접근을 하지 않습니다.

Repository: `hibou04-ops/mini-antemortem-cli` · PyPI: `mini-antemortem-cli` · import: `mini_antemortem_cli` · CLI: `mini-antemortem-cli` · MCP: `mini-antemortem-cli-mcp` with `mini-antemortem-cli[mcp]`

## 0.11.5 — 선택 bridge 계약 회귀 검증

기본 필수 의존성은 `pydantic>=2.6.0` 하나입니다. 필요한 타입·예외·계약은 내부 `_vendor` namespace에 포함하므로 `omegaprompt`, `omega-lock`, NumPy, provider SDK를 설치하지 않아도 import와 CLI가 동작합니다. `omegaprompt>=1.1.0`은 선택 extra `[bridge]`에만 있습니다. 내부 모델과 parent 모델은 별개의 Python 클래스입니다.

Python 3.11+가 필요합니다. 패키지 메타데이터와 릴리즈 태그의 대상 버전은 **0.11.5**입니다. 기본 설치 경로는 PyPI이며 bridge는 명시적으로 선택하는 extra입니다. checkout에서 개발하려면 `python -m pip install -e ".[dev,mcp]"`를 사용하세요.

```bash
pip install mini-antemortem-cli
mini-antemortem-cli --version
mini-antemortem-cli list-traps --json
python -m mini_antemortem_cli.quickstart_demo
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
