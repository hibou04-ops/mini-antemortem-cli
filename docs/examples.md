# Examples

## Deterministic Demo Replay

```bash
python examples/demo_replay.py
```

The replay uses:

- `examples/demo_config/train.jsonl`
- `examples/demo_config/test.jsonl`
- `examples/demo_config/rubric.json`
- `examples/demo_config/variants.json`
- `examples/_demo_output.txt`

It runs `mini-antemortem-cli check` through the package CLI entrypoint in text
and JSON modes. The output is deterministic and compared against the committed
fixture.

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

Expected behavior: at least one `REAL` finding and at least one `GHOST` finding.
No API keys or network access are used.

