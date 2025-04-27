# Stanford Dogs Dataset 개 품종 분류 문제

## 문제 설명 (Problem Description)
Stanford Dogs Dataset을 활용하여 개의 품종을 분류하는 딥러닝 모델을 구현하세요. 
이 데이터셋은 120개의 다른 개 품종으로 구성되어 있으며, 각 이미지는 해당 품종의 개를 포함하고 있습니다.

Implement a deep learning model to classify dog breeds using the Stanford Dogs Dataset.
This dataset consists of 120 different dog breeds, with each image containing a dog of the corresponding breed.

## 구현 요구사항 (Implementation Requirements)

### 1. 데이터 전처리 (Data Preprocessing)
- `custom_dataset.py` 파일을 생성하여 `StanfordDogsDataset` 클래스를 구현하세요.
  - `torch.utils.data.Dataset`을 상속받아야 합니다.
  - 이미지와 레이블을 로드하는 기능이 있어야 합니다.
  - 바운딩 박스 정보를 활용하여 개 부분만 크롭해야 합니다.
- `custom_dataloader.py` 파일을 생성하여 데이터 로더를 구현하세요.
  - 학습과 테스트 데이터셋을 분리해야 합니다.
  - 데이터 증강 기법을 적용해야 합니다.
  - 배치 단위로 데이터를 로드할 수 있어야 합니다.

### 2. 모델 구현 (Model Implementation)
- `custom_model.py` 파일을 생성하여 모델을 구현하세요.
  - ResNet18을 기본 모델로 사용하세요.
  - 마지막 fully connected layer를 수정하여 120개의 클래스를 분류할 수 있도록 하세요.
  - 모델의 마지막 레이어를 제외한 모든 레이어를 고정(freeze)하세요.

### 3. 학습 및 평가 (Training and Evaluation)
- `train.py` 파일을 생성하여 학습 코드를 구현하세요.
  - Adam 옵티마이저를 사용하세요.
  - Cross Entropy Loss를 손실 함수로 사용하세요.
  - 학습률(learning rate)은 0.001로 설정하세요.
  - 10 에포크 동안 학습을 진행하세요.
  - 각 에포크마다 학습 및 검증 정확도를 출력하세요.
  - 학습 과정을 시각화하세요.

## 참고 자료 (References)
1. PyTorch 공식 문서: https://pytorch.org/docs/stable/index.html
2. torchvision 모델: https://pytorch.org/vision/stable/models.html
3. Stanford Dogs Dataset: http://vision.stanford.edu/aditya86/ImageNetDogs/

## 평가 기준 (Evaluation Criteria)
- 모델 구현의 정확성 (Accuracy of model implementation)
- 코드의 가독성과 구조 (Code readability and structure)
- 학습 과정의 안정성 (Training process stability)
- 최종 모델의 성능 (Final model performance)

## 제출물 (Submission)
1. 다음 파일들을 구현하여 제출하세요:
   - `custom_dataset.py`
   - `custom_dataloader.py`
   - `custom_model.py`
   - `train.py`
2. 학습 결과 그래프 (Training result graphs)
3. 최종 모델의 정확도 (Final model accuracy)
4. 구현 과정에서의 어려움이나 특이사항 (Difficulties or notable points during implementation)

## 힌트 (Hints)
1. Dataset 클래스 구현 시 다음 메서드들이 필요합니다:
   - `__init__`: 데이터셋 초기화
   - `__len__`: 데이터셋 크기 반환
   - `__getitem__`: 인덱스에 해당하는 데이터 반환
2. DataLoader 구현 시 다음 파라미터들을 고려하세요:
   - `batch_size`
   - `shuffle`
   - `num_workers`
3. 모델 구현 시 다음을 고려하세요:
   - 사전 학습된 모델 로드
   - 레이어 수정
   - 파라미터 고정
4. 학습 코드 구현 시 다음을 고려하세요:
   - 옵티마이저 설정
   - 손실 함수 설정
   - 학습 루프 구현
   - 성능 평가 