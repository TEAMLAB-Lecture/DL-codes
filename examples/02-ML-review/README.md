# 머신러닝 기초 복습

이 디렉토리에는 머신러닝의 기본적인 모델들을 NumPy와 PyTorch로 구현한 예제들이 포함되어 있습니다.

## 파일 목록

### 선형 회귀 (Linear Regression)

1. `01_linear_regression_numpy.py`: NumPy로 구현한 선형 회귀
   - 캘리포니아 주택 가격 데이터셋 사용
   - 경사 하강법으로 모델 학습
   - 시각화: 학습 곡선, 특성 중요도, 예측값 vs 실제값

2. `02_linear_regression_pytorch.py`: PyTorch로 구현한 선형 회귀
   - 캘리포니아 주택 가격 데이터셋 사용
   - PyTorch의 자동 미분 기능 활용
   - 시각화: 학습 곡선, 특성 중요도, 예측값 vs 실제값

### 로지스틱 회귀 (Logistic Regression)

3. `03_logistic_regression_numpy.py`: NumPy로 구현한 로지스틱 회귀
   - 유방암 데이터셋 사용
   - 이진 분류 문제 해결
   - 시각화: 학습 곡선, ROC 곡선, 혼동 행렬, 특성 가중치

4. `04_logistic_regression_pytorch.py`: PyTorch로 구현한 로지스틱 회귀
   - 유방암 데이터셋 사용
   - PyTorch의 신경망 모듈 활용
   - 시각화: 학습 곡선, ROC 곡선, 혼동 행렬, 특성 가중치

## 실행 방법

각 파일은 독립적으로 실행할 수 있습니다:
```bash
# 선형 회귀
python 01_linear_regression_numpy.py
python 02_linear_regression_pytorch.py

# 로지스틱 회귀
python 03_logistic_regression_numpy.py
python 04_logistic_regression_pytorch.py
```

## 출력 파일

각 예제는 다음과 같은 시각화 파일들을 생성합니다:

### 선형 회귀
- `01_linear_regression_numpy_cost_history.png`
- `01_linear_regression_numpy_feature_importance.png`
- `01_linear_regression_numpy_prediction_vs_actual.png`
- `02_linear_regression_pytorch_cost_history.png`
- `02_linear_regression_pytorch_feature_importance.png`
- `02_linear_regression_pytorch_prediction_vs_actual.png`

### 로지스틱 회귀
- `03_logistic_regression_numpy_loss_history.png`
- `03_logistic_regression_numpy_roc_curve.png`
- `03_logistic_regression_numpy_confusion_matrix.png`
- `03_logistic_regression_numpy_feature_weights.png`
- `04_logistic_regression_pytorch_loss_history.png`
- `04_logistic_regression_pytorch_roc_curve.png`
- `04_logistic_regression_pytorch_confusion_matrix.png`
- `04_logistic_regression_pytorch_feature_weights.png` 