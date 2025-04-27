import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET

class StanfordDogsDataset(Dataset):
    """
    Stanford Dogs Dataset을 위한 Custom Dataset 클래스
    
    Args:
        root_dir (str): 데이터셋의 루트 디렉토리
        transform (callable, optional): 이미지에 적용할 변환
        train (bool, optional): True면 학습 데이터, False면 테스트 데이터
    """
    def __init__(self, root_dir, transform=None, train=True):
        # TODO: 데이터셋 초기화 코드를 구현하세요
        # 1. 이미지와 어노테이션 디렉토리 경로 설정
        # 2. 클래스 이름과 인덱스 매핑 생성
        # 3. 이미지 파일 경로와 레이블 수집
        # 4. 학습/테스트 데이터 분할
        pass
    
    def __len__(self):
        # TODO: 데이터셋의 크기를 반환하는 코드를 구현하세요
        pass
    
    def __getitem__(self, idx):
        # TODO: 인덱스에 해당하는 이미지와 레이블을 반환하는 코드를 구현하세요
        # 1. 이미지 로드
        # 2. 어노테이션 파일에서 바운딩 박스 정보 파싱
        # 3. 바운딩 박스를 사용하여 이미지 크롭
        # 4. 변환 적용
        # 5. 이미지와 레이블 반환
        pass
    
    def _parse_annotation(self, annotation_path):
        # TODO: XML 어노테이션 파일에서 바운딩 박스 정보를 파싱하는 코드를 구현하세요
        # 1. XML 파일 파싱
        # 2. 바운딩 박스 좌표 추출
        # 3. 좌표 반환
        pass 