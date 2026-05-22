# Toolkit Positioning

`mini-antemortem-cli`는 omegaprompt tool family의 작은 구성 요소입니다.

| Tool | Position |
|---|---|
| `omegaprompt` | Calibration engine. `PreflightReport`를 소비하고 adaptation plan을 도출합니다. |
| `mini-antemortem-cli` | Calibration config에 대한 deterministic analytical preflight. Provider call 없음. |
| `mini-omega-lock` | Endpoint/judge behavior에 대한 empirical live/mock preflight probe. |
| `omega-lock` | 더 넓은 optimization/audit framework. |
| `antemortem-cli` | Disk-verified citation과 source context를 다루는 pre-implementation recon CLI. |

## 실전 구분

질문이 "이 config에 알려진 구조적 calibration trap이 있는가?"라면 `mini-antemortem-cli`를 씁니다. 질문이 "이 provider path가 실제로 기대대로 동작하는가?"라면 `mini-omega-lock`을 씁니다. Repository recon과 file:line evidence가 필요하면 `antemortem-cli` 영역입니다.

이 저장소는 provider quality, prompt superiority, external validation을 claim하지 않습니다.

