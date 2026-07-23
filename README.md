# MLB 선발투수 Stuff+ 범주 예측

리그 전체에서 2021~2025년 매 시즌 `공식 선발 + 50구 이상` 등판을 20경기 이상 확보한
19명을 대상으로 다음 선발 등판의 Stuff+를 선수별 Low/Middle/High 범주로 예측합니다.

## 채택 모델

- 비선형 주 예측: TPE로 최적화한 XGBoost + EWMA4
- 해석 보조: Ridge(`alpha=100`, correction scale 0.5) + EWMA4
- 하이퍼파라미터 탐색: 2022~2024 순차 검증만 사용
- 최종 재학습: 2020~2024
- 2025 진단 평가: XGBoost 51.71%, Ridge 52.92%

XGBoost 탐색은 32개 후보를 2023년에서 10개로, 2023~2024년에서 3개로 줄인 뒤
2022~2024년 안정성 점수로 잠급니다. 동일 seed 재실행에서 같은 파라미터가 나오는 것을
확인했으며 탐색 시간은 약 29초입니다.

## XGBoost 탐색 실행

```powershell
python search_xgboost_fast.py `
  --statcast-dir data/statcast_mlb_stable_starters_2020 data/statcast_mlb_stable_starters_2021_2025 `
  --stuff data/fangraphs_stuff_mlb_stable_starters_2020.parquet data/fangraphs_stuff_mlb_stable_starters_2021_2025.parquet `
  --output-dir experiments/runs/xgboost_fast_search_2020_2025
```

## 주요 탐색 산출물

- `stage1_32_candidates_2023.csv`
- `stage2_top10_2023_2024.csv`
- `stage3_top3_2022_2024.csv`
- `locked_xgboost_config.json`
- `locked_xgboost_2025_predictions.parquet`
- `ridge_benchmark_2025_predictions.parquet`
- `tuned_xgboost_vs_ridge_2025.parquet`

전체 방법론과 결과는 `FINAL_MODEL_REPORT.md`와 `FINAL_MODEL_REPORT.html`에 정리돼 있습니다.
