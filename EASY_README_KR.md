# mini-antemortem-cli - 쉬운 설명

[README.md](README.md)의 압축 버전입니다. English easy version: [EASY_README.md](EASY_README.md). 한국어 메인 README: [README_KR.md](README_KR.md).

## 시작 · 단독 사용 · 도킹

**Omega Aile** — Quiet precision. AI research guided by evidence.

모델 호출 없이 보정 설정의 구조적 함정을 분류합니다. 보정 실행 없이 쓸 수 있지만 omegaprompt 패키지는 필수입니다.

Python 3.11 이상에서 실행합니다. 설치에는 인터넷이 필요합니다.

```bash
python -m pip install mini-antemortem-cli==0.10.1
mini-antemortem-cli --version
mini-antemortem-cli list-traps --json
```

list-traps는 내장 규칙을 출력하고 종료 0입니다. 실제 check에는 train/test JSONL·rubric·variants가 필요합니다. 기본 check는 advisory라 심각한 finding도 종료 0일 수 있습니다. --fail-on-severity는 정책 위반을 1, 입력 오류는 2로 처리합니다.

[실제 연결 방식과 입력·출력](https://github.com/hibou04-ops/omega-lock/blob/main/DOCKING.md) · [전체 안내](README.md). 두 mini 도구의 현재 검증 조합은 omegaprompt 2.1.2이며, 별도 배포 패키지입니다.

MCP는 해당 배포명의 `[mcp]` extras와 기존 서버 명령을 사용합니다. 기존 FastMCP 계약은 SDK `>=1.0.0,<2.0.0` 범위로 유지합니다.


## 이게 뭔가요?

`mini-antemortem-cli`는 프롬프트 평가 셋업에 숨은 함정(train/test 누수, judge 편향, 동질적 variant 등)이 가짜 합격 점수를 만들기 전에 잡아내는 결정론적 linter입니다. 로컬 `omegaprompt` calibration config 입력을 읽고, 11개 built-in trap pattern을 검사한 뒤 `AnalyticalFinding`을 반환합니다. API key 없음, live provider 호출 없음, network 없음. `omegaprompt`와 함께 동작하며 단독 config linter로도 유용합니다.

Repository: `hibou04-ops/mini-antemortem-cli` · PyPI: `mini-antemortem-cli` · import: `mini_antemortem_cli` · CLI: `mini-antemortem-cli` · MCP: `mini-antemortem-cli-mcp` with `mini-antemortem-cli[mcp]`

## 0.10.0의 새로운 점

- 새 trap 규칙 2개 — 이제 count는 11개(이전 9개): `few_shot_leakage_into_test`(prompt에 박힌 few-shot example이 held-out 항목과 겹침), `rubric_dead_weight_dimension`(weight 0인데 judge는 여전히 채점하는 rubric dimension).
- claim-drift 수정: trap count가 이제 모든 곳에서 일관됩니다(한 곳은 *nine*, 다른 곳은 *seven*이었음). source of truth는 `analytical_traps()`이며 consistency check가 drift 시 빌드를 실패시킵니다.
- 버전 무관 publish 워크플로: release tag, `pyproject` 버전, `__init__` 버전이 일치해야 빌드가 publish됩니다.

## 0.9.1의 새로운 점

- 잘못된 입력 오류 메시지가 이제 한 줄입니다(스키마 오류 시 여러 줄이던 것 수정). 종료 코드는 그대로 `2`.

## 0.9.0의 새로운 점

- 텍스트 출력이 이제 grep 친화적 `Summary:` 한 줄로 시작합니다(native status: PASS / ADVISORY / HOLD / BLOCK / NEEDS_MORE_EVIDENCE).
- high-signal finding이 발화한 계산값을 `cite` 필드(config-referenced)에 담습니다.
- 잘못된 입력 파일은 깔끔한 한 줄 오류로 보고하고 종료 코드 `2`로 종료합니다.
- `list-traps --json`으로 `{id, hypothesis}` 배열을 출력합니다.
- train/test ID 정확 겹침은 이제 `BLOCKER`로 발화합니다(이전 high). `--fail-on-severity high`는 여전히 이를 잡습니다.
- `4 - Beta`로 이동, 이후 surface는 additive-only.

## 설치

```bash
pip install mini-antemortem-cli
```

MCP 사용:

```bash
pip install "mini-antemortem-cli[mcp]"
mini-antemortem-cli-mcp
```

## Trap IDs

- `self_agreement_bias`
- `small_sample_kc4_power`
- `variants_homogeneous`
- `rubric_weight_concentration`
- `judge_budget_too_small`
- `empty_reference_with_strict_rubric`
- `no_held_out_slice`
- `train_test_id_overlap`
- `routed_provider_opaque_family`
- `few_shot_leakage_into_test`
- `rubric_dead_weight_dimension`

소스 기반 count와 hypothesis는 여기에서 생성됩니다: [docs/generated/claims_kr.md](docs/generated/claims_kr.md).

## 빠른 CLI 데모

```bash
python examples/demo_replay.py
```

직접 CLI 실행:

```bash
mini-antemortem-cli check \
  --target-provider openai \
  --target-model gpt-4o \
  --judge-provider openai \
  --judge-model gpt-4o \
  --train examples/demo_config/train.jsonl \
  --test examples/demo_config/test.jsonl \
  --rubric examples/demo_config/rubric.json \
  --variants examples/demo_config/variants.json \
  --json
```

## Python API

```python
from mini_antemortem_cli import analytical_preflight, analytical_traps
```

Calibration 전에 `analytical_preflight(...)`를 실행하고, 결과를 `omegaprompt.preflight.PreflightReport`와 `derive_adaptation_plan`에 넘기면 됩니다.

## 쓸 때

- Provider call 비용을 쓰기 전에 deterministic sanity check가 필요할 때.
- CI에서 `--fail-on-severity high`로 구조적 위험을 막고 싶을 때.
- 시간에 따라 diff 가능한 explicit trap ID가 필요할 때.

## 다른 도구를 쓸 때

- Live/mock provider probe가 필요하면 `mini-omega-lock`.
- Source-file recon과 disk-verified file:line citation이 필요하면 `antemortem-cli`.
- 실제 calibration engine은 `omegaprompt`.

License: Apache 2.0.
