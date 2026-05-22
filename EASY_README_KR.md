# mini-antemortem-cli - 쉬운 설명

[README.md](README.md)의 압축 버전입니다. English easy version: [EASY_README.md](EASY_README.md).

## 이게 뭔가요?

`mini-antemortem-cli`는 `omegaprompt` calibration config를 위한 deterministic preflight checker입니다. 로컬 config 입력을 읽고, 9개 built-in trap pattern을 검사한 뒤 `AnalyticalFinding`을 반환합니다. API key 없음, live provider 호출 없음, network 없음.

Repository: `hibou04-ops/mini-antemortem-cli` · PyPI: `mini-antemortem-cli` · import: `mini_antemortem_cli` · CLI: `mini-antemortem-cli` · MCP: `mini-antemortem-cli-mcp` with `mini-antemortem-cli[mcp]`

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
