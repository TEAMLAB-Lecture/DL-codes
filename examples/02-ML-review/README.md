# 선형 회귀 예제: 캘리포니아 주택 가격 예측

이 예제는 캘리포니아 주택 가격 데이터셋을 사용하여 다중 선형 회귀 모델을 구현합니다. NumPy와 PyTorch 두 가지 버전으로 구현되어 있습니다.

## 데이터
- 캘리포니아 주택 가격 데이터셋 (California Housing Dataset)
- 8개의 특성:
  1. MedInc: 지역의 중간 소득
  2. HouseAge: 주택의 중간 연령
  3. AveRooms: 평균 방 개수
  4. AveBedrms: 평균 침실 개수
  5. Population: 인구
  6. AveOccup: 평균 점유율
  7. Latitude: 위도
  8. Longitude: 경도
- 타겟: 주택 가격 (10만 달러 단위)

## 구현된 기능
1. 경사 하강법을 사용한 선형 회귀 모델 구현
   - NumPy 버전: 수동으로 경사 하강법 구현
   - PyTorch 버전: 자동 미분과 PyTorch의 nn.Module 활용
2. 데이터 전처리
   - 학습/테스트 데이터 분할
   - 특성 스케일링
3. 모델 평가
   - MSE (Mean Squared Error)
   - R² Score
4. 시각화
   - Cost/Loss 변화 그래프
   - 특성 중요도 시각화
   - 예측값 vs 실제값 산점도
5. PyTorch 특화 기능
   - 자동 미분 (AutoGrad)
   - nn.Sequential을 사용한 고급 모델 구현
   - 배치 처리

## 실행 방법
1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 코드 실행:
```bash
# NumPy 버전
python linear_regression_example.py

# PyTorch 버전
python pytorch_linear_regression.py
```

## 출력
- 데이터 정보 및 기술 통계
- 학습 과정에서의 비용과 가중치 변화
- 모델 성능 평가 지표 (MSE, R²)
- 특성 중요도 분석
- 새로운 주택에 대한 가격 예측
- 시각화된 그래프들
  - NumPy 버전: cost_history.png, feature_importance.png, prediction_vs_actual.png
  - PyTorch 버전: pytorch_cost_history.png, pytorch_feature_importance.png, pytorch_prediction_vs_actual.png 