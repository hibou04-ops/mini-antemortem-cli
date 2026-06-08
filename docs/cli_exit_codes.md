# CLI Exit Codes

`mini-antemortem-cli check` is advisory by default.

| Situation | Exit code | Notes |
|---|---:|---|
| Normal success | 0 | Findings are printed, including `REAL` findings, unless a fail-on gate is configured. |
| Policy gate failure | 1 | Triggered by `--fail-on-severity LEVEL` when selected labels meet or exceed `LEVEL`. Default labels are `REAL,UNRESOLVED`. |
| Usage/configuration error | 2 | Raised by `argparse` for missing required arguments or invalid flag values. As of 0.9.0, input-file load failures (missing file, malformed JSON, schema mismatch) are caught and reported as a one-line stderr message naming the file and error class, then exit `2` — distinct from the policy-gate exit `1`. |
| Deprecated `--fail-on-blocker` | 0 or 1 | Alias for blocker-level gating. Current built-in traps do not normally emit `blocker`. Prefer `--fail-on-severity high`. |
| JSON output | Same as policy result | `--json` changes output format only; it does not disable fail-on behavior. |

Example CI gate:

```bash
mini-antemortem-cli check \
  --target-provider openai \
  --judge-provider anthropic \
  --train train.jsonl \
  --test test.jsonl \
  --rubric rubric.json \
  --variants variants.json \
  --json \
  --fail-on-severity high
```

