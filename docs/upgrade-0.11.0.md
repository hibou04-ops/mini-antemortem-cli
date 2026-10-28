# 0.11.0 검증과 이행

기준은 `291dd9178ae31bad0e20e8fb3b8893263ab52bd7`(공개 0.10.1과 같은 commit)입니다. Windows, Python 3.12, omegaprompt 2.1.2 격리 환경에서 실제 domain 객체를 생성하여 비교했습니다. provider 호출은 없습니다.

| 개발 사례 | 이전 | 수정 후 |
|---|---|---|
| 실제 입력 중복·입력/정답 쌍 중복 2개 | 2개 탐지 | 2개 탐지 |
| 다른 입력의 공유 라벨 yes/positive/0 | 3개 오탐 | 0개 오탐 |
| 대소문자가 다른 식별자 | 1개 오탐 | 0개 오탐 |
| 단순 어휘 겹침 | 정상 | 정상 |
| 1,000개 같은 prompt의 최대 Jaccard | 1.0 | 1.0 |
| 같은 작업의 token 정규화 수 | 1,000 | 1 |
| 같은 작업의 union/intersection 수 | 각각 499,500 | 각각 0 |

7개 개발 사례의 정답은 3/7에서 7/7로 바뀌었습니다. 이 사례는 알려진 결함을 재현한 자료이며 일반 오탐률·미탐률, 블라인드 결과, 실제 모델 품질을 뜻하지 않습니다. 기존 정상 corpus/golden 판정은 그대로 검사합니다. 반복 prompt 작업의 연산 수만 개선 근거로 사용하며 전체 시간 배속을 주장하지 않습니다. 전부 다른 prompt에서는 여전히 정확한 최대값을 위해 이차 비교가 필요할 수 있습니다.

```bash
python benchmarks/precision_cost.py
python benchmarks/precision_cost.py --source-root /path/to/baseline
python -m pytest -q tests/test_precision_contracts.py
python scripts/run_false_positive_audit.py --check
```

[이전](../benchmarks/upgrade-0.11.0/before.json), [수정 후](../benchmarks/upgrade-0.11.0/after.json) 결과를 제공합니다. 기존 golden fixture의 판정은 바꾸지 않았습니다. demo 문구는 output-label 판정 범위를 정정하여 생성기로 갱신했고 그 변경에 한해 integrity manifest를 갱신했습니다. 기존 잘못된 output-only 기대 테스트는 다른 입력의 공통 라벨이 실제 입력 노출을 입증하지 못한다는 이유로 교정했습니다.

입력 비교는 Unicode NFC와 공백을 정규화하지만 대소문자를 보존합니다. 구조화 helper는 dict 순서와 타입을 보존합니다. 현재 `PromptVariants`의 공개 모델은 few-shot 값을 문자열로 제한하므로 숫자/구조화 값 혼동을 공개 생성 경로의 재현 결함으로 계산하지 않았습니다.

`selected_few_shot_indices`가 생략되면 pool의 잠재 입력 노출을 검사합니다. 선택된 실행을 검사하려면 실제 pool 인덱스를 넘기세요(빈 목록도 가능). matching input은 위험으로 남으며 output 라벨만 공유하는 것은 누출로 판정하지 않습니다. 의미상 유사 입력과 answer-only disclosure는 탐지 범위 밖입니다.

`TrapPolicy.from_json_file`은 이제 미지 키와 잘못된 타입·NaN·Inf·범위 밖 비율을 입력 오류로 거절합니다. `fail_on_missing_test=False`는 no-held-out trap만 끄며 독립 평가를 생성하거나 측정하지 않습니다.

`validation_mode='auto'|'paired'|'disjoint'`는 API/CLI/MCP에 연결됩니다. auto/disjoint의 겹친 ID는 기존 BLOCKER입니다. paired에서도 ID 집합이 완전히 같고 각 ID의 input/reference가 같을 때만 재측정 `UNRESOLVED/MEDIUM`입니다. 일부 겹침·내용 충돌·중복 ID는 면제하지 않습니다. 이 경로는 일반화의 증거가 아닙니다.

ID를 바꿔도 train/test의 같은 input은 모든 mode에서 BLOCKER입니다. reference만 공유하는 분류 라벨은 이 비교에 포함하지 않습니다. 기존 within-slice duplicate 테스트가 실제로 양쪽 모두 `input=x`를 사용해 내용도 겹쳤으므로 새 판정은 BLOCKER로 교정했습니다. 내용이 다른 slice의 중복 ID는 계속 MEDIUM인 별도 회귀검사를 추가했습니다. 기존 golden/정상 corpus 판정은 이 강화 후에도 보존됩니다.

버전은 판정 의미와 입력 오류 정책이 바뀌어 minor인 0.11.0입니다. CLI/MCP tool 이름과 기존 finding 필드는 유지합니다. CLI는 기본 0(출력 완료), 명시 gate 차단 1, 설정 오류 2를 구분합니다. PyPI 발행은 기존 수동 workflow의 정확한 `release_tag`로 실행해야 하며 GitHub Release만으로 발행되지 않습니다.
