# 불펜 투수 등판 리스크 3범주 분류 모델

불펜 투수를 특정 경기에서 기용해도 되는지 판단하기 위해, 등판 단위 성과를 본인 baseline 대비 `하/중/상` 3개 범주로 분류하는 프로젝트입니다.

현재 베이스라인 모델은 하나입니다.

```text
classification_residual_tertile_xgboost
```

## 판단 기준

라벨은 `residual = baseline_skill - shrunk_xwOBA`를 기준으로 만듭니다.

```text
하/risk   = train residual 하위 1/3
중/normal = train residual 가운데 1/3
상/good   = train residual 상위 1/3
```

xwOBA는 낮을수록 좋은 투구 결과이므로, `residual`이 낮으면 본인 평소보다 나쁜 등판, 높으면 본인 평소보다 좋은 등판입니다.

## 구조

```text
data/                  # 캐시 데이터, git ignore
lib/
  collect_statcast.py  # Statcast 수집
  build_outings.py     # pitch-level -> 등판 단위 집계
  data_prep.py         # 공통 피처 엔지니어링
  labeling.py          # baseline, shrunk_xwOBA, residual 생성
  evaluate.py          # 3범주 분류 평가/로그
  sanity_checks.py     # leakage/라벨 분포 검사
models/
  model_classification_residual_tertile_xgboost.py
compare.py             # 현재 분류 베이스라인 실행
experiments/
  experiments_log.csv  # 현재 베이스라인 실험 로그
  runs/                # 실행 산출물, git ignore
```

## 빠른 실행

```bash
pip install -r requirements.txt
python compare.py --demo
```

`--demo`는 실제 Statcast 없이 synthetic outing 데이터를 생성해서 피처, 라벨, 모델 실행, 실험 로그까지 end-to-end로 확인합니다.

## MLB 5년치 베이스라인 실행

```bash
python compare.py ^
  --input data/outings_mlb_bullpen_2021_2025.parquet ^
  --run-id mlb_bullpen_2021_2025_min30ip100bf_3class_baseline ^
  --output-dir experiments/runs/mlb_bullpen_2021_2025_min30ip100bf_3class_baseline ^
  --min-pitcher-ip 30 ^
  --min-pitcher-bf 100
```

`--models`를 생략하면 기본값으로 `classification_residual_tertile_xgboost`만 실행합니다.

## 입력 피처

현재 baseline feature set은 `personalized_workload_max`입니다.

포함:

- workload 원값: `acute_workload_7d`, `chronic_workload_28d`, `ACWR`, `rest_days`, `back_to_back`
- standard abuse rolling feature
- 투수별 workload/abuse z-score: `*_pitcher_z`
- Statcast tracking rolling feature: `*_ma5`, `*_slope5`, `*_z`
- pitch mix rolling feature: `pitch_mix_*_ma5`, `pitch_mix_*_slope5`, `pitch_mix_*_z`
- `pitcher_prior_outing_count`

제외:

- `role`: 현재 데이터에서 구조적 의미가 약해 baseline에서 제외
- `baseline_skill`, `shrunk_xwOBA`, `outing_xwOBA`, `residual`, `target_y`: 결과/라벨 계열이라 입력 피처로 사용하지 않음

## 평가 지표

```text
accuracy
balanced_accuracy
macro_f1
risk_precision
risk_recall
top20_risk_lift
```

실전 의사결정에서는 단순 accuracy보다 `risk_precision`, `risk_recall`, `top20_risk_lift`를 더 중요하게 봅니다. 특히 `proba_risk` 상위권에 실제 하/risk 등판이 얼마나 농축되는지가 핵심입니다.

## 해석 주의

이 프로젝트의 결과는 상관관계 기반 예측입니다. 감독의 기용 결정, 경기 상황, 선수 상태 같은 선택편향이 워크로드와 성과 모두에 영향을 줄 수 있으므로 인과효과로 해석하면 안 됩니다.
