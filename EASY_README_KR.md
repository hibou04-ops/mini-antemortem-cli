# mini-antemortem-cli — 쉬운 설명

> 본 README가 어렵게 느껴지는 분들을 위한 압축 버전.
> 원본: [README_KR.md](README_KR.md) · English easy: [EASY_README.md](EASY_README.md)

## 이게 뭔가요?

[omegaprompt](https://pypi.org/project/omegaprompt/)용 선택적 플러그인. Calibration 실행 전에 **config를 sanity-check** — 7개 calibration-특화 trap 패턴 대조. Pure deterministic rules. API 호출 0. 네트워크 0. ~1 ms 에 완료.

자매 [mini-omega-lock](https://pypi.org/project/mini-omega-lock/) 은 live API로 *환경*을 측정. 이건 *config*만 읽고 구조적으로 unsafe한 지점을 알려줌.

## 설치

```bash
pip install mini-antemortem-cli
```

`omegaprompt>=1.1.0` 필요 (preflight contract schema import).

## 7개 trap classifier

각각은 config 에 대한 순수 함수. `REAL` (규칙 발동), `GHOST` (해당 축에서 config 안전), `NEW` (엣지 케이스), `UNRESOLVED` 리턴.

| # | Trap id | 구체적 규칙 | 발동 시 severity |
|---|---|---|---|
| 1 | `self_agreement_bias` | target provider **와** model == judge provider, model | HIGH (둘 다 일치) / MEDIUM (same provider, 다른 model) |
| 2 | `small_sample_kc4_power` | test size < 10, 또는 train+test < 20 | HIGH (< 10) / MEDIUM (< 20 total) |
| 3 | `variants_homogeneous` | prompt < 2개, 또는 prompt ≤3개이면서 길이 범위 < 20 chars | MEDIUM |
| 4 | `rubric_weight_concentration` | 단일 dimension weight > 70% | MEDIUM |
| 5 | `judge_budget_too_small` | budget = SMALL **이면서** (dimensions + gates) > 5 | MEDIUM |
| 6 | `empty_reference_with_strict_rubric` | 어떤 item도 `reference` 필드 없음 | LOW |
| 7 | `no_held_out_slice` | test dataset 미제공 | HIGH |

모든 finding에 구체적 `note` (어느 규칙이 발동) + `remediation` (뭘 할지) 포함. `BLOCKER` severity는 guarded run을 abort; `HIGH`는 자동 AdaptationPlan override 발동 (예: homogeneous variants 일 때 sensitivity axis 건너뛰기).

## 최소 동작 예제

```python
from omegaprompt.domain.dataset import Dataset, DatasetItem
from omegaprompt.domain.judge import Dimension, HardGate, JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.preflight import PreflightReport, derive_adaptation_plan
from mini_antemortem_cli import analytical_preflight

rubric = JudgeRubric(
    dimensions=[
        Dimension(name="accuracy", description="correct", weight=0.85),
        Dimension(name="clarity",  description="readable", weight=0.15),
    ],
    hard_gates=[HardGate(name="no_refusal", description="...", evaluator="judge")],
)
variants = PromptVariants(system_prompts=["You are an assistant."], few_shot_examples=[])
train = Dataset(items=[DatasetItem(id=f"t{i}", input=f"task {i}") for i in range(5)])
test  = Dataset(items=[DatasetItem(id=f"v{i}", input=f"val {i}") for i in range(3)])

findings = analytical_preflight(
    target_provider="openai", target_model="gpt-4o-mini",
    judge_provider="openai",  judge_model="gpt-4o-mini",   # <-- same vendor + same model = trap #1 발동
    train_dataset=train,
    test_dataset=test,
    rubric=rubric,                                          # <-- 한 dim 에 weight 0.85 = trap #4 발동
    variants=variants,                                      # <-- 단일 prompt = trap #3 발동
    judge_output_budget="small",
)

for f in findings:
    print(f"{f.trap_id}: {f.label} ({f.severity}) — {f.note}")

# omegaprompt adaptation layer 에 feed:
report = PreflightReport(analytical_findings=findings)
plan   = derive_adaptation_plan(report=report)
# plan.skip_axes, plan.max_gap_override 등
```

**네트워크 없음.** **API 키 없음.** **같은 config → 항상 같은 findings.**

## 4개 export

```python
from mini_antemortem_cli import (
    analytical_preflight,   # 복합 — 7개 classifier 모두 실행
    analytical_traps,       # 7개 TrapPattern 정의 tuple 리턴
    CALIBRATION_TRAPS,      # tuple 자체 (조회용)
    TrapPattern,            # dataclass: id + hypothesis
)
```

## 쓸 때

- 실제 calibration 실행 직전에 1ms gut-check 원함.
- 구조적 trap 있는 config를 CI가 차단하게 하고 싶음 (예: target과 judge 에 같은 model 실수로 설정).
- Config 변경 간 diff 가능한 deterministic finding 원함.

## 건너뛸 때

- Config 이미 sane하다고 확신 (동일 프로젝트, 몇 주간 미변경).
- 빠른 반복 중이라 1ms 마찰도 싫음.

## mini-omega-lock 과 합성

둘 다 같은 `PreflightReport` 에 plug in:

```python
from mini_omega_lock import empirical_preflight
from mini_antemortem_cli import analytical_preflight

jq, ep, perf = empirical_preflight(...)           # live API 측정
findings      = analytical_preflight(...)          # deterministic 규칙

report = PreflightReport(
    judge_quality=jq, endpoint=ep, performance=perf,
    analytical_findings=findings,
)
plan = derive_adaptation_plan(report=report)
```

Empirical 은 환경이 *실제로 뭘 하는지* 측정. Analytical 은 config 가 *구조적으로 뭘 위험에 빠뜨리는지* 포착. 보완적, 중복 아님.

## 더 깊이

- Classifier 구현: `src/mini_antemortem_cli/traps.py`
- Finding contract: `omegaprompt.preflight.contracts.AnalyticalFinding`
- Severity → AdaptationPlan 매핑: `omegaprompt.preflight.adaptation.derive_adaptation_plan`
- 자매 empirical preflight (live API probe 포함): [mini-omega-lock](https://pypi.org/project/mini-omega-lock/)

License: Apache 2.0. Copyright (c) 2026 hibou.
