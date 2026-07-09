# 3범주 분류 파이프라인 진단

## 현재 기준

회귀 모델은 폐기하고, 등판을 `하/중/상`으로 나누는 3범주 분류를 베이스라인으로 둔다.

```text
model: classification_residual_tertile_xgboost
feature_set: personalized_workload_max
target: residual_class
```

## 라벨 해석

```text
residual = baseline_skill - shrunk_xwOBA
```

xwOBA는 낮을수록 좋으므로:

```text
residual 낮음 -> 본인 기준보다 나쁜 등판 -> 하/risk
residual 중간 -> 본인 기준 보통 등판     -> 중/normal
residual 높음 -> 본인 기준보다 좋은 등판 -> 상/good
```

## 현재 성능 진단

MLB 2021-2025, 30 IP 이상 + 100 BF 이상 표본 결과:

```text
balanced_accuracy 0.365988
macro_f1          0.353410
risk_precision    0.401074
risk_recall       0.184059
top20_risk_lift   1.207408
```

해석:

```text
무작위 3분류보다는 약간 낫다.
하지만 강한 판별기는 아니다.
현재는 실제 하/risk 등판을 넓게 잡아내기보다, 위험 후보를 약하게 농축하는 수준이다.
```

## 혼동행렬 요약

```text
actual 하/risk, predicted 하/risk     448
actual 하/risk, predicted 중/normal  1165
actual 하/risk, predicted 상/good     821

actual 중/normal, predicted 하/risk   380
actual 중/normal, predicted 중/normal 1435
actual 중/normal, predicted 상/good   898

actual 상/good, predicted 하/risk     289
actual 상/good, predicted 중/normal  1184
actual 상/good, predicted 상/good     922
```

가장 큰 문제는 실제 하/risk 등판 중 상당수가 `중` 또는 `상`으로 빠진다는 점이다. 그래서 다음 개선은 accuracy가 아니라 `risk_recall`과 `top20_risk_lift`를 올리는 방향이어야 한다.

## 유지할 원칙

- 현재 등판 결과는 입력 피처로 사용하지 않는다.
- 컷포인트는 train 구간에서만 계산한다.
- 투수 개인 baseline과 workload z-score는 모두 prior-only로 계산한다.
- `role`은 현재 baseline에서 제외한다.
- 평가 기준은 rolling-origin split을 기본으로 둔다.

## 다음 개선 후보

1. `risk` recall을 직접 높이는 threshold 정책

   현재 `argmax(proba)`로 클래스를 고른다. 실전 의사결정에서는 `proba_risk`가 일정 기준 이상이면 `하/risk`로 보내는 정책이 더 맞을 수 있다.

2. cost-sensitive 평가

   `상/good`을 `하/risk`로 오판하는 것과 `하/risk`를 `상/good`으로 오판하는 것은 비용이 다르다. 다음 단계에서는 class별 비용을 명시해야 한다.

3. season walk-forward 검증

   단일 rolling-origin split만으로는 안정성이 부족하다. 최종 보고용 검증은 season walk-forward CV로 확장한다.

4. 후보군 랭킹 평가

   실제 운영에서는 모든 투수의 정확한 클래스를 맞히기보다, 오늘 피해야 할 투수 후보를 위로 올리는 것이 중요하다. 따라서 `topK_risk_rate`, `topK_lift`를 중심 지표로 추가한다.
