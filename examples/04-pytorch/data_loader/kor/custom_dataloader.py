import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import StanfordDogsDataset

def get_dataloaders(root_dir, batch_size=32, num_workers=4):
    """
    학습 및 테스트 DataLoader를 생성합니다.
    
    Args:
        root_dir (str): 데이터셋의 루트 디렉토리
        batch_size (int): 배치 크기
        num_workers (int): 데이터 로딩에 사용할 워커 수
        
    Returns:
        tuple: (train_loader, test_loader, num_classes)
    """
    # TODO: 데이터 증강 및 전처리 변환을 정의하세요
    # 1. 학습 데이터용 변환 (증강 포함)
    #   - 크기 조정
    #   - 랜덤 수평 반전
    #   - 랜덤 회전
    #   - 색상 조정
    #   - 텐서 변환
    #   - 정규화
    train_transform = transforms.Compose([
        # TODO: 학습 데이터용 변환을 구현하세요
    ])
    
    # 2. 테스트 데이터용 변환 (기본 전처리만)
    #   - 크기 조정
    #   - 텐서 변환
    #   - 정규화
    test_transform = transforms.Compose([
        # TODO: 테스트 데이터용 변환을 구현하세요
    ])
    
    # TODO: 데이터셋을 생성하세요
    # 1. 학습 데이터셋 생성
    train_dataset = StanfordDogsDataset(
        # TODO: 파라미터를 설정하세요
    )
    
    # 2. 테스트 데이터셋 생성
    test_dataset = StanfordDogsDataset(
        # TODO: 파라미터를 설정하세요
    )
    
    # TODO: DataLoader를 생성하세요
    # 1. 학습용 DataLoader
    train_loader = DataLoader(
        # TODO: 파라미터를 설정하세요
    )
    
    # 2. 테스트용 DataLoader
    test_loader = DataLoader(
        # TODO: 파라미터를 설정하세요
    )
    
    # TODO: 클래스 수를 가져오세요
    num_classes = # TODO: 클래스 수를 설정하세요
    
    return train_loader, test_loader, num_classes 