# 머신러닝 리뷰 예제

이 폴더에는 머신러닝의 기본 개념들을 리뷰하기 위한 예제 코드들이 포함되어 있습니다.

## 1. 선형 회귀 예제
- `linear_regression_example.py`: NumPy를 사용한 선형 회귀 구현
- `pytorch_linear_regression.py`: PyTorch를 사용한 선형 회귀 구현
- 캘리포니아 주택 가격 데이터셋을 사용한 다중 선형 회귀 문제
- 경사 하강법을 통한 모델 학습
- 모델 성능 평가 및 시각화

## 2. 로지스틱 회귀 예제
- `logistic_regression_example.py`: PyTorch를 사용한 로지스틱 회귀 구현
- 당뇨병 예측을 위한 이진 분류 문제
- 8개의 특성을 가진 가상 데이터셋 사용:
  1. age (나이)
  2. bmi (체질량지수)
  3. glucose (혈당)
  4. bp (혈압)
  5. insulin (인슐린)
  6. family_hist (가족력)
  7. physical_act (신체활동)
  8. smoking (흡연)
- 구현된 기능:
  - 데이터 전처리 및 정규화
  - 미니배치 학습
  - 모델 성능 평가 (정확도, 정밀도, 재현율, F1 스코어)
  - 혼동 행렬 시각화
  - 특성 중요도 분석
  - 결정 경계 시각화 (PCA 사용)
  - 새로운 환자 데이터에 대한 예측
  - 로그 오즈비 해석

## 실행 방법
1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 예제 실행:
```bash
# 선형 회귀 (NumPy 버전)
python linear_regression_example.py

# 선형 회귀 (PyTorch 버전)
python pytorch_linear_regression.py

# 로지스틱 회귀
python logistic_regression_example.py
```

## 출력 파일
### 선형 회귀
- `cost_history.png`: 학습 과정에서의 비용 변화
- `feature_importance.png`: 특성 중요도 시각화
- `prediction_vs_actual.png`: 예측값 vs 실제값 비교

### 로지스틱 회귀
- `logistic_regression_learning_curves.png`: 학습 및 테스트 손실/정확도 변화
- `logistic_regression_confusion_matrix.png`: 혼동 행렬
- `logistic_regression_weights.png`: 학습된 가중치 시각화
- `logistic_regression_decision_boundary.png`: PCA를 사용한 결정 경계 시각화 