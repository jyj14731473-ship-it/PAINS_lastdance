# MLB 선발투수 Stuff+ 범주 예측 최종 종합 보고서

- 작성일: 2026-07-24
- 주 예측 모델: `xgboost_tpe_span4_compact_locked`
- 해석 보조 모델: `ridge_span4_compact_alpha100_scale0.5`
- 개발 데이터: 2020~2024년
- 잠긴 테스트: 2025년
- 분석 단위: 투수별 50구 이상 공식 선발 등판
- 최종 평가 표본: 19명, 497등판

## 1. 요약

2025를 후보 선택에서 차단한 시간 순서 파이프라인으로 XGBoost와 Ridge 두 모델을 채택했다.
XGBoost는 32→10→3 단계 TPE 탐색으로 잠갔고, Ridge는 Ridge 계열 검증 1위 설정을
해석 가능한 보조 모델로 사용한다.

| 역할 | 모델 | 2025 정확도 | 균형 정확도 | Macro F1 | 순서 MAE |
|---|---|---:|---:|---:|---:|
| 주 예측 | XGBoost | 51.71% | 49.31% | 0.496 | 0.547 |
| 해석 보조 | Ridge | **52.92%** | **51.05%** | **0.514** | **0.531** |

두 모델은 497건 중 455건에서 같은 범주를 예측해 일치율 91.55%를 기록했다. 일치한
등판의 정확도는 53.41%였다. 불일치 42건에서는 XGBoost가 33.33%, Ridge가 47.62%를
맞혔다. 이 결과는 설명용 진단이며 2025를 보고 모델 설정이나 역할을 변경하지 않았다.

실무적으로는 XGBoost의 비선형 예측과 Ridge의 해석을 함께 제공하되, 두 모델이 다르면
불확실성 경고를 붙이는 방식이 적절하다. 현재 성능은 다음 등판의 세부 변동을 강하게
예측하는 수준이 아니라, 최근 Stuff+ 상태를 EWMA로 안정화하고 작은 잔차 보정을 더하는
수준으로 해석해야 한다.

## 2. 시간 분리와 채택 원칙

이번 실행은 다음 순서를 강제한다.

1. 모델 선택 함수에는 2024년 이하 데이터만 전달한다.
2. 2020~2021 학습 → 2022 검증을 수행한다.
3. 2020~2022 학습 → 2023 검증을 수행한다.
4. 2020~2023 학습 → 2024 검증을 수행한다.
5. 평균 균형 정확도에서 연도별 표준편차의 절반을 뺀 점수로 XGBoost를 확정한다.
6. Ridge보다 평균 균형 정확도가 0.5%p 이상 높아야 XGBoost 주 모델 조건을 통과한다.
7. 두 모델을 2020~2024년으로 재학습한 뒤 2025년에 적용한다.

최종 XGBoost는 Ridge보다 검증 평균 균형 정확도가 0.517%p 높아 기준을 간신히 통과했다.
두 모델의 하이퍼파라미터는 모두 2025 성능과 무관하게 정해졌다.

## 3. 데이터

### 3.1 출처

| 데이터 | 사용 내용 |
|---|---|
| Statcast | 투구 수, 구속, 회전수, 릴리스, 무브먼트, 구종 구성 |
| FanGraphs 경기 로그 | 공식 선발 여부와 선발 Stuff+ |
| MLB 공식 투구 기록 | 선수 식별자와 이름 |

2020년은 단축 시즌이므로 선수 자격 판정에는 사용하지 않고 추가 학습 이력으로만 사용했다.

### 3.2 분석 대상

2021~2025년의 모든 시즌에서 `공식 선발 + Statcast 50구 이상` 등판을 20회 이상 확보한
19명을 대상으로 했다. 2025 등판 수는 테스트 모집단을 확정하는 데 사용하지만, 2025
Stuff+는 후보 또는 하이퍼파라미터 선택에 사용하지 않았다.

### 3.3 시즌별 표본

| 시즌 | 50구 이상 선발 | 완전사례 | 결측 제외 |
|---:|---:|---:|---:|
| 2020 | 176 | 118 | 58 |
| 2021 | 512 | 436 | 76 |
| 2022 | 546 | 482 | 64 |
| 2023 | 583 | 520 | 63 |
| 2024 | 570 | 502 | 68 |
| 2025 | 560 | 497 | 63 |

입력 또는 타깃에 결측이 있는 행은 학습과 평가에서 제외했다. 두 채택 모델과 모든 기준선은
동일한 2025년 497등판에서 비교했다.

## 4. 타깃

각 투수의 학습 구간 Stuff+에서 33.3%와 66.7% 분위수를 계산했다.

- Low: Stuff+ ≤ 학습 구간 33.3% 분위수
- Middle: 두 분위수 사이
- High: Stuff+ > 학습 구간 66.7% 분위수

검증 폴드는 해당 검증연도 이전의 학습 자료만으로 경계값을 만든다. 최종 2025 테스트는
2020~2024년 Stuff+만으로 선수별 경계값을 만든다.

## 5. 피처

### 5.1 공통 후보 28개

워크로드:

- `prev_start_pitch_count`
- `rest_days`
- `workload_density_3starts`

Stuff+ 이력:

- `prior_stuff_plus`
- `stuff_plus_mean_last5`
- `stuff_plus_slope_last5`

물리 피처:

- `release_speed_ma5`, `release_speed_slope5`
- `release_spin_rate_ma5`, `release_spin_rate_slope5`
- `release_extension_ma5`, `release_extension_slope5`
- `release_pos_x_ma5`, `release_pos_x_slope5`
- `release_pos_z_ma5`, `release_pos_z_slope5`
- `arm_angle_ma5`, `arm_angle_slope5`
- `pfx_x_ma5`, `pfx_x_slope5`
- `pfx_z_ma5`, `pfx_z_slope5`
- `spin_axis_sin_ma5`, `spin_axis_cos_ma5`

구종 구성:

- `breaking_share_ma5`, `breaking_share_slope5`
- `offspeed_share_ma5`, `offspeed_share_slope5`

구종은 패스트볼·브레이킹볼·오프스피드로 묶었다. 합계 제약으로 인한 중복을 피하기 위해
패스트볼 비중은 직접 입력하지 않았다.

### 5.2 채택 모델의 compact 16개 피처

두 채택 모델은 동일한 compact 피처를 사용한다.

- `prior_minus_ewma4`
- `mean5_minus_ewma4`
- `stuff_plus_slope_last5`
- `prev_start_pitch_count`
- `rest_days`
- `workload_density_3starts`
- `release_speed_slope5`
- `release_spin_rate_slope5`
- `release_extension_slope5`
- `release_pos_x_slope5`
- `release_pos_z_slope5`
- `arm_angle_slope5`
- `pfx_x_slope5`
- `pfx_z_slope5`
- `breaking_share_slope5`
- `offspeed_share_slope5`

## 6. 모델 구조

두 모델 모두 span 4 EWMA를 기본 예측값으로 두고 현재 Stuff+와 EWMA 사이의 잔차를
예측한다.

```text
residual = current_stuff_plus - prior_EWMA4
predicted_stuff = prior_EWMA4 + correction_scale × predicted_residual
```

### 6.1 XGBoost

- 역할: 비선형 주 예측
- feature set: compact 16개
- EWMA span: 4
- trees: 100
- max depth: 2
- learning rate: 0.04221
- min child weight: 15.043
- L2(`reg_lambda`): 16.823
- L1(`reg_alpha`): 0.706
- gamma: 0
- subsample: 0.9997
- column sample: 0.5856
- correction scale: 0.5750
- tree method: histogram, max bins 64

### 6.2 Ridge

- 역할: 해석 가능한 보조 예측
- feature set: compact 16개
- EWMA span: 4
- alpha: 100
- correction scale: 0.5
- 입력 피처: 투수별 학습 데이터에서 표준화

Ridge의 alpha가 크고 잔차도 50%만 반영하므로 과격한 보정보다 EWMA4 주변의 보수적인
선형 조정을 수행한다.

## 7. 검증 결과와 모델 채택

XGBoost는 다음 단계로 탐색했다.

1. TPE 후보 32개를 2023년에 평가하고 상위 10개 유지
2. 상위 10개를 2023~2024년에 평가하고 상위 3개 유지
3. 상위 3개를 2022~2024년에 평가
4. 최고 안정성 점수에서 0.3%p 이내면 더 단순한 설정 선택

선택 점수는 `평균 균형 정확도 - 0.5 × 연도별 표준편차`다.

| 모델 | 2022 | 2023 | 2024 | 평균 | 표준편차 | 선택 점수 |
|---|---:|---:|---:|---:|---:|---:|
| 최적화 XGBoost | 54.27% | 53.57% | 55.83% | **54.56%** | **0.94%p** | **54.09%** |
| Ridge | 53.39% | 52.98% | 55.76% | 54.04% | 1.23%p | 53.43% |
| EWMA4 | 54.65% | 51.99% | 55.67% | 54.10% | 1.55%p | 53.33% |

XGBoost의 Ridge 대비 평균 균형 정확도 우위는 0.517%p로 사전에 정한 0.5%p 조건을
통과했다. 동일 seed로 전체 탐색을 두 번 실행했을 때 잠금 파라미터가 완전히 같았고,
각 실행 시간은 약 29초였다.

## 8. 2025 테스트

| 모델 | N | 정확도 | 균형 정확도 | Macro F1 | 순서 MAE |
|---|---:|---:|---:|---:|---:|
| **Ridge 해석 보조** | 497 | **52.92%** | **51.05%** | **0.514** | **0.531** |
| EWMA4 단독 | 497 | 52.31% | 50.65% | 0.510 | 0.545 |
| XGBoost 주 예측 | 497 | 51.71% | 49.31% | 0.496 | 0.547 |
| 직전 Stuff+ | 497 | 49.90% | 47.94% | 0.480 | 0.610 |
| 학습 최빈 범주 | 497 | 42.86% | 33.33% | 0.200 | 0.819 |

Ridge가 XGBoost보다 6건 더 맞혔고 EWMA4보다는 3건 더 맞혔다. 최적화 XGBoost는 기존
고정 XGBoost보다 4건을 추가로 맞혔다. 테스트 결과는 두 모델의
설명과 비교에 사용하되, 이 결과를 근거로 하이퍼파라미터를 변경하지 않았다.

### 8.1 XGBoost 혼동행렬

| 실제＼예측 | Low | Middle | High |
|---|---:|---:|---:|
| Low | **129** | 70 | 14 |
| Middle | 35 | **87** | 39 |
| High | 18 | 64 | **41** |

### 8.2 Ridge 혼동행렬

| 실제＼예측 | Low | Middle | High |
|---|---:|---:|---:|
| Low | **128** | 70 | 15 |
| Middle | 36 | **87** | 38 |
| High | 15 | 60 | **48** |

Ridge는 XGBoost보다 High를 7건 더 맞혔고 Low는 1건, Middle은 같았다.

## 9. 두 모델의 일치와 활용

| 구분 | 등판 수 | 비율 또는 정확도 |
|---|---:|---:|
| 같은 범주 예측 | 455 | 91.55% |
| 다른 범주 예측 | 42 | 8.45% |
| 일치 구간 정확도 | 455 | 53.41% |
| 불일치 구간 XGBoost 정확도 | 42 | 33.33% |
| 불일치 구간 Ridge 정확도 | 42 | 47.62% |

권장 출력 방식은 다음과 같다.

- 두 모델의 범주가 같으면 `agreement=True`로 함께 제시한다.
- 다르면 예측 불확실성이 높은 등판으로 표시한다.
- XGBoost 결과에는 Ridge 결과와 주요 계수 방향을 함께 제공한다.
- 현재 데이터만으로 두 모델의 평균이나 투표 규칙을 새로 조정하지 않는다.

## 10. 선수별 2025 결과

| 투수 | N | XGBoost | Ridge | 두 모델 일치율 |
|---|---:|---:|---:|---:|
| José Berríos | 27 | 92.6% | 92.6% | 100.0% |
| Mitch Keller | 29 | 82.8% | 82.8% | 100.0% |
| Zac Gallen | 30 | 76.7% | 73.3% | 96.7% |
| Luis Castillo | 29 | 75.9% | 69.0% | 93.1% |
| Kyle Freeland | 25 | 60.0% | 64.0% | 88.0% |
| Kevin Gausman | 29 | 55.2% | 58.6% | 89.7% |
| Zack Wheeler | 21 | 57.1% | 57.1% | 100.0% |
| Brady Singer | 29 | 55.2% | 51.7% | 93.1% |
| Charlie Morton | 22 | 45.5% | 50.0% | 90.9% |
| Michael Wacha | 27 | 44.4% | 48.1% | 96.3% |
| Tyler Anderson | 23 | 56.5% | 47.8% | 91.3% |
| Jameson Taillon | 17 | 41.2% | 47.1% | 88.2% |
| Logan Gilbert | 18 | 38.9% | 44.4% | 94.4% |
| Sonny Gray | 28 | 32.1% | 42.9% | 85.7% |
| Logan Webb | 31 | 35.5% | 38.7% | 90.3% |
| Chris Bassitt | 28 | 42.9% | 35.7% | 92.9% |
| Dylan Cease | 29 | 24.1% | 34.5% | 82.8% |
| Patrick Corbin | 27 | 22.2% | 33.3% | 88.9% |
| Framber Valdez | 28 | 35.7% | 28.6% | 78.6% |

선수별 편차가 매우 크다. 상위 선수에서는 한 시즌의 범주가 특정 상태에 오래 머문 반면,
하위 선수에서는 2025 분포 변화와 경기별 변동을 두 모델 모두 충분히 설명하지 못했다.
따라서 전체 정확도를 모든 투수에게 동일하게 적용되는 품질로 해석하면 안 된다.

## 11. Ridge 전역 해석

Ridge 계수는 각 투수의 학습 데이터에서 표준화된 피처에 대한 잔차 보정 계수이며, 최종
반영률 0.5까지 적용한 값이다. 절댓값 중앙값이 큰 피처는 다음과 같다.

| 피처 | 계수 중앙값 | 절댓값 중앙값 | 양수 선수 비율 | 해석 |
|---|---:|---:|---:|---|
| `stuff_plus_slope_last5` | -0.189 | 0.189 | 0.0% | 최근 상승·하락 추세의 평균회귀 보정 |
| `prior_minus_ewma4` | -0.138 | 0.138 | 15.8% | 직전 경기의 EWMA 이탈을 반대 방향으로 조정 |
| `release_spin_rate_slope5` | -0.001 | 0.128 | 47.4% | 영향 크기는 있으나 선수별 방향이 다름 |
| `mean5_minus_ewma4` | +0.119 | 0.119 | 94.7% | 최근 5경기 평균이 EWMA보다 높으면 양의 보정 |
| `arm_angle_slope5` | +0.039 | 0.113 | 57.9% | 선수별 이질성이 큰 보조 신호 |
| `prev_start_pitch_count` | -0.041 | 0.107 | 31.6% | 직전 투구 수 증가가 대체로 약한 음의 보정 |
| `release_speed_slope5` | -0.016 | 0.107 | 36.8% | 구속 변화의 방향은 선수별로 다름 |

가장 일관된 신호는 Stuff+ 자체의 평균회귀다. `stuff_plus_slope_last5`는 19명 모두 음수였고,
`mean5_minus_ewma4`는 19명 중 18명에서 양수였다. 반면 회전수·암 슬롯·구속 변화는 절댓값이
커도 부호가 엇갈리므로 전 리그에 동일한 인과 방향이 있다고 해석하면 안 된다.

## 12. 한계

1. 2025 한 시즌만 시간 외 테스트로 사용했다.
2. 투수별 모델이라 각 모델의 학습 표본이 작다.
3. 완전사례 정책으로 2025년 560등판 중 63등판을 제외했다.
4. 2021~2025 매년 20선발을 충족한 생존자 표본이다.
5. 2025 등판 수로 테스트 모집단을 사후 확정했다.
6. 새 코드에서는 2025를 선택 함수에서 차단했지만 연구자가 이전 탐색에서 2025를 본 사실은
   소급해 제거할 수 없다.
7. Ridge를 보조 모델로 함께 채택하는 결정은 첫 XGBoost 테스트 확인 이후 이루어졌다.
   Ridge 설정 자체는 초기 2023·2024 검증만으로 자동 확정했지만, Ridge의 2025 성능은 보조적
   기술 결과로 해석하는 것이 안전하다.

## 13. 재현 방법

```powershell
.\.venv\Scripts\python.exe search_xgboost_fast.py `
  --statcast-dir data/statcast_mlb_stable_starters_2020 data/statcast_mlb_stable_starters_2021_2025 `
  --stuff data/fangraphs_stuff_mlb_stable_starters_2020.parquet data/fangraphs_stuff_mlb_stable_starters_2021_2025.parquet `
  --output-dir experiments/runs/xgboost_fast_search_2020_2025

.\.venv\Scripts\python.exe stuff_mlb_temporal_final.py `
  --statcast-dir data/statcast_mlb_stable_starters_2020 data/statcast_mlb_stable_starters_2021_2025 `
  --stuff data/fangraphs_stuff_mlb_stable_starters_2020.parquet data/fangraphs_stuff_mlb_stable_starters_2021_2025.parquet `
  --official-stats data/mlb_official_pitching_2021_2025.parquet `
  --output-dir experiments/runs/mlb_stuff_temporal_selected_2020_2025
```

| 파일 | 내용 |
|---|---|
| `locked_xgboost_config.json` | 결정론적으로 잠근 XGBoost 파라미터와 검증·진단 성능 |
| `stage1_32_candidates_2023.csv` | 1단계 TPE 후보 |
| `stage2_top10_2023_2024.csv` | 2단계 생존 후보 |
| `stage3_top3_2022_2024.csv` | 최종 3개 후보 |
| `locked_xgboost_2025_predictions.parquet` | 잠금 XGBoost의 2025 예측 |
| `tuned_xgboost_vs_ridge_2025.parquet` | 최적화 XGBoost와 Ridge 경기별 비교 |
| `validation_candidates.csv` | 2025 제외 후보 222개의 검증 성능과 순위 |
| `pooled_metrics.csv` | 두 모델과 기준선의 2025 전체 성능 |
| `per_pitcher_metrics.csv` | 선수별 모델 성능 |
| `dual_model_comparison.parquet` | 경기별 두 모델 예측과 일치 여부 |
| `ridge_coefficients.csv` | 19명 × 16피처 표준화 계수 |
| `ridge_feature_summary.csv` | 계수 중앙값·사분위수·부호 일관성 |
| `confusion_matrix_xgboost.csv` | XGBoost 혼동행렬 |
| `confusion_matrix_ridge.csv` | Ridge 혼동행렬 |
| `predictions.parquet` | 경기별 모든 예측 |
| `report.json` | 선택 규칙, 이중 모델 설정, 성능과 일치율 |

## 14. 결론

XGBoost와 Ridge는 2025 이전 검증만으로 설정을 확정했다. 최적화 XGBoost는 Ridge보다
검증 평균 균형 정확도가 0.517%p 높아 주 모델 조건을 통과했다. 2025에서는 Ridge가
정확도 52.92%로 XGBoost의 51.71%보다 높았고, 두 모델은 91.55%의 등판에서 같은 범주를
냈다. 따라서 XGBoost의 비선형 예측과 Ridge의 안정적·해석 가능한
보조 예측을 함께 제공한다. 다음 시즌에는 두 모델과 일치 규칙을 변경하지 않은 상태로
전향 평가해야 한다.

최종적으로 이 모델의 실질적인 기반은 과거 Stuff+의 지속성을 반영한 EWMA4다. XGBoost와
Ridge는 그 주변의 잔차만 보정하며, 2025에서 가장 좋은 Ridge도 EWMA4보다 3경기를 더
맞힌 수준이다. 그러므로 복잡한 모델 자체보다 시간 분리, 강한 규제, 이중 모델 일치 표시,
선수별 성능 편차를 함께 관리하는 것이 더 중요하다.
