# 다층 퍼셉트론(MLP) 예제 모음

이 폴더는 다층 퍼셉트론(MLP)의 다양한 구현 예제를 포함합니다.

## 1. Fashion MNIST MLP 구현

Fashion MNIST 데이터셋을 사용한 이미지 분류 MLP 구현입니다.

### 주요 특징
- 3층 MLP 구조 (입력층, 2개의 은닉층, 출력층)
- 드롭아웃을 통한 과적합 방지
- 학습 과정 시각화
- 혼동 행렬을 통한 성능 분석

### 파일
- `fashion_mnist_mlp.py`: Python 스크립트 버전
- `fashion_mnist_mlp.ipynb`: Jupyter Notebook 버전

## 2. IRIS 데이터셋 MLP 구현

IRIS 데이터셋을 사용한 꽃 종류 분류 MLP 구현입니다.

### 주요 특징
- Softmax 회귀를 사용한 다중 클래스 분류
- PyTorch를 활용한 구현
- 학습 과정 시각화
- 결정 경계 시각화

### 파일
- `iris-with-softmax.py`: Softmax 회귀를 사용한 Python 스크립트 버전
- `iris-with-softmax.ipynb`: Softmax 회귀를 사용한 Jupyter Notebook 버전
- `iris-with-pytorch.py`: PyTorch를 사용한 Python 스크립트 버전
- `iris-with-pytorch.ipynb`: PyTorch를 사용한 Jupyter Notebook 버전

## 실행 방법

각 예제를 실행하기 위한 방법은 다음과 같습니다:

### Fashion MNIST MLP
1. 필요한 패키지 설치:
```bash
pip install torch torchvision matplotlib numpy scikit-learn seaborn
```

2. 코드 실행:
```bash
python fashion_mnist_mlp.py
```

### IRIS MLP
1. 필요한 패키지 설치:
```bash
pip install torch numpy matplotlib scikit-learn pandas
```

2. 코드 실행:
```bash
# Softmax 버전
python iris-with-softmax.py

# PyTorch 버전
python iris-with-pytorch.py
```

## 예상 결과

### Fashion MNIST MLP
- 테스트 정확도: 약 85-90%
- 학습/테스트 손실 및 정확도 그래프
- 혼동 행렬 시각화
- 예측 결과 시각화

### IRIS MLP
- 테스트 정확도: 약 95-100%
- 학습 과정 시각화
- 결정 경계 시각화
- 클래스별 분류 성능 분석

## 참고사항

- GPU가 사용 가능한 경우 자동으로 GPU를 사용합니다.
- 학습된 모델은 각각의 파일에 저장됩니다.
- 시각화 결과는 matplotlib을 통해 표시됩니다. 