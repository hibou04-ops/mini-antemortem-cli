# Examples

## Deterministic Demo Replay

```bash
python examples/demo_replay.py
```

Replay가 사용하는 fixture:

- `examples/demo_config/train.jsonl`
- `examples/demo_config/test.jsonl`
- `examples/demo_config/rubric.json`
- `examples/demo_config/variants.json`
- `examples/_demo_output.txt`

Script는 package CLI entrypoint를 통해 `mini-antemortem-cli check`를 text mode와 JSON mode로 실행합니다. Output은 deterministic이며 committed fixture와 비교됩니다.

## Direct CLI

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
  --judge-output-budget small \
  --json
```

Expected behavior: 최소 하나의 `REAL` finding과 최소 하나의 `GHOST` finding이 나옵니다. API key와 network access는 사용하지 않습니다.

