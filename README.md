# 불펜 투수 워크로드-성과 회귀 모델

불펜 투수의 등판 단위 기대 성과를 예측하고, 워크로드가 선수의 평소 컨디션 대비 성과에 어떤 영향을 주는지 비교 실험하는 프로젝트입니다.

예측값 `target_y`는 0~1 범위입니다.

- `0.5`: 평소 실력 수준
- `1.0`에 가까움: 평소보다 호투
- `0.0`에 가까움: 평소보다 부진

## 구조

```text
data/                  # 캐시 데이터, git ignore
lib/
  collect_statcast.py  # Statcast 수집
  build_outings.py     # pitch-level -> 등판 단위 집계
  data_prep.py         # 공통 피처 엔지니어링
  labeling.py          # baseline 대비 target_y 생성
  evaluate.py          # 공통 평가/로그/플롯
  sanity_checks.py     # leakage/라벨 분포 검사
models/
  model_xgboost.py
  model_gam_raw.py
  model_gam_acwr.py
  model_form_momentum_ewm.py    # 과거 target_y 최근폼 EWM 앙상블 (현재 최고 성능)
  model_form_momentum_blend.py  # EWM 앙상블 + 모멘텀 ridge 50/50 블렌드
compare.py             # 모든 model_*.py 자동 실행/비교
experiments/
  experiments_log.csv  # 실험 로그, git 추적
  runs/                # 실행 산출물, git ignore
```

모든 `.py` 파일은 VSCode Python Interactive Window에서 셀 단위 실행할 수 있도록 `# %%` 구분자를 포함합니다.

## 빠른 실행

```bash
pip install -r requirements.txt
python compare.py --demo
```

`--demo`는 실제 Statcast 없이 synthetic outing 데이터를 생성해서 피처, 라벨, 모델 비교, 실험 로그까지 end-to-end로 확인합니다.

## 실제 데이터 흐름

```bash
python -m lib.collect_statcast --start 2021-03-01 --end 2024-10-01 --out-dir data/statcast
python -m lib.build_outings --input-dir data/statcast --output data/outings.parquet
python compare.py --input data/outings.parquet
```

불펜만 분석하려면 `--relief-only`로 1회에 등판을 시작한 선발 등판을 제외합니다.

```bash
python -m lib.build_outings --input-dir data/statcast_lad_2025_regular --output data/outings_relief.parquet --relief-only
python compare.py --input data/outings_relief.parquet
```

`compare.py`는 입력 데이터에 `target_y`가 없으면 `lib.data_prep.prepare_features()`와 `lib.labeling.create_labels()`를 먼저 실행합니다.

## 누수 방지 원칙

- 워크로드, 물리 트렌드, baseline은 모두 현재 등판 이전 데이터만 사용합니다.
- rolling/expanding 계열 계산은 현재 행을 제외합니다.
- `outing_xwOBA`, `shrunk_xwOBA`, `baseline_skill`, `target_y` 같은 결과/라벨 계열은 모델 입력에서 제외됩니다.
- 예외적으로 `lib/momentum_features.py`의 `prior_y_*` 피처는 **이전 등판들의** `target_y`를
  shift(1)로만 사용합니다. 예측 시점에는 이미 끝난 등판의 결과이므로 누수가 아니며,
  현재 등판의 라벨은 절대 피처에 들어가지 않습니다.
- `sanity_checks.py`가 `feature_asof_date <= game_date`와 normal-condition 라벨 평균을 검사합니다.

## 모델링 발견 (LAD 2025 불펜, 593 등판)

- 상수 0.5 예측의 테스트 RMSE는 0.2703인데, 워크로드 피처만 쓰는 기존 모델은 전부
  이를 넘지 못했습니다 (최고 0.2756). 워크로드 단독 상관은 train/test 간 부호가 뒤집힐
  정도로 약합니다.
- 반면 target_y는 선수 내 lag-1 자기상관이 약 +0.46으로, **과거 target 이력(최근 폼)**이
  가장 강한 예측 신호입니다. 이를 사용하는 `form_momentum_ewm`이 테스트 RMSE 0.2430으로
  기존 최고 대비 약 12% 개선했습니다.
- 하이퍼파라미터는 train 내부 forward-chaining CV로만 선택했고, 단일 설정 선택 대신
  (halflife, shrink_k) 그리드 평균(하이퍼파라미터 앙상블)으로 선택 분산을 줄였습니다.

## 새 방법론 추가

`models/model_new_method.py` 파일 하나를 추가하고 아래 인터페이스만 구현하면 됩니다.

```python
def run(config: dict, train_df, test_df) -> dict:
    return {
        "model_name": "...",
        "config": config,
        "metrics": {"rmse": ..., "mae": ..., "rmse_low_bf": ...},
        "predictions": ...,
        "model_object": ...,
        "git_commit": "...",
    }
```

`compare.py`는 `models/model_*.py`를 자동 탐색하므로 기존 모델 파일을 수정할 필요가 없습니다.

## 해석 주의

이 프로젝트의 결과는 상관관계 분석입니다. 감독의 기용 결정, 경기 상황, 선수 상태 같은 선택편향이 워크로드와 성과 모두에 영향을 줄 수 있으므로 인과효과로 해석하면 안 됩니다.
