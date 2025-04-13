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
│   ├── 02-ML-review/              # 머신러닝 기초 복습
│   │   ├── README.md              # 디렉토리 설명
│   │   ├── 01_linear_regression_numpy.py    # NumPy로 구현한 선형 회귀
│   │   ├── 02_linear_regression_pytorch.py  # PyTorch로 구현한 선형 회귀
│   │   ├── 03_logistic_regression_numpy.py  # NumPy로 구현한 로지스틱 회귀
│   │   └── 04_logistic_regression_pytorch.py # PyTorch로 구현한 로지스틱 회귀
│   │
│   └── 03-MLP/                    # 다층 퍼셉트론
│       ├── README.md              # 디렉토리 설명
│       ├── fashion_mnist_mlp.py   # Fashion MNIST MLP 구현
│       ├── fashion_mnist_mlp.ipynb # Fashion MNIST MLP Jupyter Notebook
│       ├── iris-with-softmax.py   # IRIS 데이터셋 Softmax 구현
│       ├── iris-with-softmax.ipynb # IRIS 데이터셋 Softmax Jupyter Notebook
│       ├── iris-with-pytorch.py   # IRIS 데이터셋 PyTorch 구현
│       └── iris-with-pytorch.ipynb # IRIS 데이터셋 PyTorch Jupyter Notebook
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
python examples/02-ML-review/01_linear_regression_numpy.py
```
- PyTorch 버전:
```bash
python examples/02-ML-review/02_linear_regression_pytorch.py
```

#### 로지스틱 회귀
- NumPy 버전:
```bash
python examples/02-ML-review/03_logistic_regression_numpy.py
```
- PyTorch 버전:
```bash
python examples/02-ML-review/04_logistic_regression_pytorch.py
```

### 3. 다층 퍼셉트론 (examples/03-MLP/)

#### Fashion MNIST MLP
```bash
python examples/03-MLP/fashion_mnist_mlp.py
```

#### IRIS 데이터셋 MLP
- Softmax 버전:
```bash
python examples/03-MLP/iris-with-softmax.py
```
- PyTorch 버전:
```bash
python examples/03-MLP/iris-with-pytorch.py
```

## 필요한 패키지

이 프로젝트를 실행하기 위해 필요한 주요 패키지들입니다:

```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install torch
pip install torchvision
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

# Deep Learning Course Environment

이 프로젝트는 딥러닝 강좌를 위한 Docker 기반 개발 환경을 제공합니다.

## 사전 요구사항

- Docker Desktop
- NVIDIA GPU (선택사항, GPU 가속을 위해 필요)

## 설치 방법

### Windows 사용자

1. Docker Desktop 설치
   - [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop) 다운로드 및 설치
   - 설치 후 Docker Desktop 실행

2. NVIDIA GPU 사용자 (선택사항)
   - [NVIDIA Driver](https://www.nvidia.com/download/index.aspx) 설치
   - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) 설치

### Mac/Linux 사용자

1. Docker 설치
   - Mac: [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop) 설치
   - Linux: 
     ```bash
     curl -fsSL https://get.docker.com -o get-docker.sh
     sudo sh get-docker.sh
     ```

2. NVIDIA GPU 사용자 (Linux만 해당)
   - [NVIDIA Driver](https://www.nvidia.com/download/index.aspx) 설치
   - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) 설치

## 실행 방법

### Windows 사용자

1. `start.bat` 파일을 더블클릭하거나 명령 프롬프트에서 실행:
   ```cmd
   start.bat
   ```

### Mac/Linux 사용자

1. 스크립트에 실행 권한 부여:
   ```bash
   chmod +x start.sh
   ```

2. 스크립트 실행:
   ```bash
   ./start.sh
   ```

## 접속 방법

- Jupyter Notebook이 시작되면 웹 브라우저에서 http://localhost:8888 접속
- 토큰은 터미널에 표시됩니다

## 주의사항

- Docker Desktop이 실행 중이어야 합니다
- GPU 사용 시 NVIDIA 드라이버가 올바르게 설치되어 있어야 합니다
- 컨테이너 종료는 `Ctrl+C`를 누르거나 터미널 창을 닫으면 됩니다

## 문제 해결

- Docker가 실행되지 않은 경우: Docker Desktop을 실행한 후 다시 시도하세요
- GPU 관련 문제: NVIDIA 드라이버와 Container Toolkit이 올바르게 설치되어 있는지 확인하세요
- 포트 충돌: 8888 포트가 사용 중인 경우 다른 포트를 사용하도록 설정을 변경할 수 있습니다 