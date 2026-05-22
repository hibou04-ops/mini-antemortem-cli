# Toolkit Positioning

`mini-antemortem-cli` is one small component in the omegaprompt tool family.

| Tool | Position |
|---|---|
| `omegaprompt` | Calibration engine. It consumes `PreflightReport` and derives adaptation plans. |
| `mini-antemortem-cli` | Deterministic analytical preflight over calibration config. No provider calls. |
| `mini-omega-lock` | Empirical live/mock preflight probes for endpoint and judge behavior. |
| `omega-lock` | Broader optimization and audit framework. |
| `antemortem-cli` | Pre-implementation recon CLI with disk-verified citations and broader code/source context. |

## Practical Split

Use `mini-antemortem-cli` when the question is "does this config contain a
known structural calibration trap?" Use `mini-omega-lock` when the question is
"does this provider path behave as expected?" Use `antemortem-cli` when the
question requires repository recon and file:line evidence.

This repository does not claim provider quality, prompt superiority, or external
validation.

