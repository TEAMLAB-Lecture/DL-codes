# Deep Learning Class Codes

이 저장소는 딥러닝 수업에서 사용되는 코드 예제들을 포함하고 있습니다.

## 디렉토리 구조

```
.
├── examples/
│   ├── 01-intro-to-neural-networks/ # 신경망 입문
│   │   ├── README.md              # 디렉토리 설명
│   │   └── pytroch_example_codes.ipynb  # PyTorch 기초 예제
│   │
│   └── 02-ML-review/              # 머신러닝 기초 복습
│       ├── README.md              # 디렉토리 설명
│       ├── 01_linear_regression_numpy.py    # NumPy로 구현한 선형 회귀
│       ├── 02_linear_regression_pytorch.py  # PyTorch로 구현한 선형 회귀
│       ├── 03_logistic_regression_numpy.py  # NumPy로 구현한 로지스틱 회귀
│       └── 04_logistic_regression_pytorch.py # PyTorch로 구현한 로지스틱 회귀
│
└── README.md                      # 이 파일
```

## 예제 실행 방법

### 1. 신경망 입문 (examples/01-intro-to-neural-networks/)

Jupyter 노트북을 실행하려면:
```bash
jupyter notebook examples/01-intro-to-neural-networks/pytroch_example_codes.ipynb
```

### 2. 머신러닝 기초 복습 (examples/02-ML-review/)

#### 선형 회귀
- NumPy 버전:
```bash
python examples/03-ML-review/01_linear_regression_numpy.py
```
- PyTorch 버전:
```bash
python examples/03-ML-review/02_linear_regression_pytorch.py
```

#### 로지스틱 회귀
- NumPy 버전:
```bash
python examples/03-ML-review/03_logistic_regression_numpy.py
```
- PyTorch 버전:
```bash
python examples/03-ML-review/04_logistic_regression_pytorch.py
```

## 필요한 패키지

이 프로젝트를 실행하기 위해 필요한 주요 패키지들입니다:

```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install torch
pip install seaborn
pip install jupyter
```

## 주의사항

1. 한글 폰트 설정
   - matplotlib에서 한글을 표시하기 위해 나눔고딕 폰트가 필요합니다.
   - Windows의 경우: `C:\Windows\Fonts\NanumGothic.ttf`
   - Linux의 경우: `/usr/share/fonts/truetype/nanum/NanumGothic.ttf`

2. CUDA 지원
   - PyTorch 예제들은 CUDA가 설치된 경우 자동으로 GPU를 사용합니다.
   - GPU 사용이 필요한 경우 CUDA와 cuDNN이 설치되어 있어야 합니다.

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 