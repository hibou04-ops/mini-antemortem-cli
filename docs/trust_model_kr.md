# Trust Model

`mini-antemortem-cli`는 `omegaprompt` calibration config의 구조적 속성을 검증합니다. Model quality 자체를 검증하지는 않습니다.

## 검증하는 것

- `analytical_traps()`가 노출하는 source-backed built-in trap registry.
- 로컬 dataset/rubric/variant 입력에 대한 deterministic classification.
- 모든 `AnalyticalFinding`의 explicit trap ID.
- 0.9.0부터, high-signal trap에 대한 config-referenced citation(`cite` 필드):
  trap이 발화한 계산값(겹치는 id, 지배적 rubric dimension과 weight,
  variant의 최대 pairwise Jaccard, 너무 작은 test slice). 이는 사용자가
  제공한 calibration config를 가리키며, 디스크상의 소스 파일이 아닙니다.
- 로컬 classifier rule에 따른 severity assignment.
- `--fail-on-severity`, `--fail-on-label` 기반 CLI policy gate.
- MCP filesystem input의 workspace boundary.
- `omegaprompt.preflight.PreflightReport`, `derive_adaptation_plan`과의 shape compatibility.

Current trap IDs:

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

## 검증하지 않는 것

- Provider/model superiority.
- Calibration result의 statistical validity.
- Production adoption 또는 external validation.
- Live endpoint reliability.
- Disk-verified source file citation. (0.9.0의 `cite` 필드는 디스크상의 소스
  파일이 아니라 제공된 calibration config를 가리킵니다.)
- Append-only audit trail.

## Deterministic / No-Network 경계

Default tests, demo replay, golden cases, fixture integrity, release audit, publish readiness는 provider call과 network 없이 실행되도록 설계되어 있습니다. 패키지는 로컬 파일 또는 inline object만 읽습니다.

## Trap Policy Threshold 경계

`TrapPolicy`는 minimum held-out size, near-duplicate Jaccard, rubric concentration, small-budget axis limit 같은 threshold를 조정합니다. Policy override는 classification threshold를 바꾸지만 trap registry 자체를 바꾸지 않습니다.

## Train/Test Split Caveat

이 패키지는 missing held-out slice와 train/test ID overlap을 잡을 수 있습니다. 하지만 held-out slice가 대표성, 독립성, domain별 충분한 크기를 가진다는 점은 증명하지 않습니다.

## Same-Vendor / Routed-Provider 한계

같은 family의 target/judge pair는 self-agreement risk로 flag됩니다. Routed provider는 served-model family가 static config pass에서 보이지 않으므로 `UNRESOLVED`로 flag됩니다.

## Severity 의미

- `blocker`: emit되면 hard stop.
- `high`: CI gate와 adaptation-plan discipline에 충분히 강한 signal.
- `medium`: actionable warning.
- `low`: informational 또는 clean-path evidence.

## CLI Fail-On Policy

기본 `check`는 finding을 출력하고 exit `0`입니다. `--fail-on-severity`를 주면 `--fail-on-label`로 선택한 label에 대해 policy gate가 됩니다. 기본 label은 `REAL,UNRESOLVED`입니다.

## MCP Boundary

MCP server는 `analytical_preflight`, `list_traps`를 노출합니다. Path input은 `MINI_ANTEMORTEM_WORKSPACE_ROOT` 또는 현재 working directory 안으로 제한됩니다. Inline dict/list input은 filesystem을 읽지 않습니다.

## Adaptation-Plan Compatibility

Finding은 `omegaprompt.preflight.contracts.AnalyticalFinding` object로 emit됩니다. 이 패키지는 `AdaptationPlan`을 직접 mutate하지 않고, `omegaprompt`가 소비할 수 있는 input을 제공합니다.
