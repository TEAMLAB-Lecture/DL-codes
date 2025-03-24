# 머신러닝 기초 복습

이 디렉토리에서는 머신러닝의 기본적인 개념들을 복습하고 구현해봅니다.

## 1. 선형 회귀 예제

### NumPy 버전 (`linear_regression_example.py`)
- 캘리포니아 주택 가격 예측 문제
- 데이터셋: sklearn의 California Housing Dataset
- 특성:
  - MedInc: 지역의 중간 소득
  - HouseAge: 주택 연령
  - AveRooms: 평균 방 개수
  - AveBedrms: 평균 침실 개수
  - Population: 인구
  - AveOccup: 평균 점유율
  - Latitude: 위도
  - Longitude: 경도
- 구현된 기능:
  - 경사 하강법을 사용한 선형 회귀 모델 구현
  - 데이터 전처리 (정규화)
  - 모델 평가 (MSE, R² Score)
  - 학습 과정 시각화
  - 특성 중요도 분석

### PyTorch 버전 (`pytorch_linear_regression.py`)
- NumPy 버전과 동일한 문제를 PyTorch로 구현
- 추가 기능:
  - 자동 미분 (AutoGrad) 활용
  - nn.Sequential을 사용한 모델 구현
  - GPU 지원 (가능한 경우)
  - 배치 처리

## 2. 로지스틱 회귀 예제

### NumPy 버전 (`numpy_logistic_regression.py`)
- 당뇨병 예측 문제
- 데이터셋: sklearn의 Diabetes Dataset
- 특성:
  - age: 나이
  - sex: 성별
  - bmi: 체질량 지수
  - bp: 혈압
  - s1-s6: 혈액 검사 결과
- 구현된 기능:
  - 경사 하강법을 사용한 로지스틱 회귀 모델 구현
  - 특성 엔지니어링 (BMI와 혈압의 상호작용, BMI 제곱)
  - 데이터 전처리 (정규화)
  - 교차 검증
  - 모델 평가 (정확도, 정밀도, 재현율, F1, ROC-AUC)
  - 다양한 시각화:
    - 학습 곡선
    - 혼동 행렬
    - ROC 커브
    - 특성 중요도
    - 결정 경계 (PCA 사용)
    - 수학적 개념 시각화 (시그모이드, 로그 오즈, BCE 손실)
  - 모델 해석:
    - 특성 중요도 분석
    - 로그 오즈비 해석
    - 다양한 임계값에 따른 성능 평가

### PyTorch 버전 (`pytorch_logistic_regression.py`)
- NumPy 버전과 동일한 문제를 PyTorch로 구현
- 추가 기능:
  - 자동 미분 (AutoGrad) 활용
  - nn.Sequential을 사용한 모델 구현
  - GPU 지원 (가능한 경우)
  - 배치 처리
  - DataLoader를 사용한 데이터 관리

## 실행 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 예제 실행:
```bash
# NumPy 버전 선형 회귀
python linear_regression_example.py

# PyTorch 버전 선형 회귀
python pytorch_linear_regression.py

# NumPy 버전 로지스틱 회귀
python numpy_logistic_regression.py

# PyTorch 버전 로지스틱 회귀
python pytorch_logistic_regression.py
```

## 출력 파일

각 예제는 다음과 같은 시각화 파일들을 생성합니다:

### 선형 회귀
- `linear_regression_learning_curves.png`: 학습 과정의 손실과 정확도 변화
- `linear_regression_predictions.png`: 실제값과 예측값 비교
- `linear_regression_feature_importance.png`: 특성 중요도 분석

### 로지스틱 회귀
- `logistic_regression_learning_curves.png`: 학습 과정의 손실과 정확도 변화
- `logistic_regression_confusion_matrix.png`: 혼동 행렬
- `logistic_regression_roc_curve.png`: ROC 커브
- `logistic_regression_weights.png`: 학습된 가중치 시각화
- `logistic_regression_decision_boundary.png`: PCA를 사용한 결정 경계
- `logistic_regression_concept_visualization.png`: 로지스틱 회귀의 수학적 개념 시각화 