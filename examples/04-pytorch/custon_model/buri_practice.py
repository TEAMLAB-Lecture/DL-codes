"""
BURI 모델 연습 문제

이 파일은 BURI 모델의 기본 구조와 주요 기능을 연습하기 위한 문제들을 포함하고 있습니다.
각 문제는 점진적으로 난이도가 높아지도록 구성되어 있습니다.

문제 1: 기본 BURI 모델 구현
- BURIModel 클래스를 완성하세요.
- 입력 크기, 은닉층 크기, 출력 크기를 파라미터로 받는 모델을 구현하세요.
- 3개의 완전 연결 레이어를 사용하고, ReLU 활성화 함수를 적용하세요.
- 적절히 Batch Normalization을 적용하세요.

문제 2: 모델 학습 함수 구현
- train_model 함수를 완성하세요.
- 배치 단위로 데이터를 처리하고, 손실값을 계산하세요.
- 옵티마이저를 사용하여 모델 파라미터를 업데이트하세요.
- 100번째 배치마다 손실값을 출력하세요.

문제 3: 메인 함수 구현
- main 함수에서 모델 파라미터를 설정하세요.
- MNIST 데이터셋을 위한 적절한 파라미터 값을 선택하세요.
- 모델 인스턴스를 생성하고, 손실 함수와 옵티마이저를 설정하세요.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 문제 1: BURIModel 클래스 구현
class BURIModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # TODO: 모델 초기화 코드를 작성하세요
        pass
        
    def forward(self, x):
        # TODO: 순전파 함수를 작성하세요
        pass

# 문제 2: 모델 학습 함수 구현
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    # TODO: 모델 학습 코드를 작성하세요
    pass

# 문제 3: 메인 함수 구현
def main():
    # TODO: 모델 파라미터 설정
    input_size = None  # MNIST 이미지 크기
    hidden_size = None
    output_size = None  # 0-9 숫자 분류
    
    # TODO: 모델 생성 및 학습 설정
    pass

if __name__ == "__main__":
    main() 