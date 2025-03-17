# Deep Learning Tutorials with Docker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository provides a Docker-based development environment for deep learning tutorials. It includes ready-to-run examples for neural networks using both TensorFlow and PyTorch.

## 🚀 Features

- 📦 Pre-configured Docker environment
- 🔧 GPU support (NVIDIA CUDA)
- 📓 Interactive Jupyter notebooks
- 🎓 Step-by-step tutorials
- 🔄 Both TensorFlow and PyTorch examples

## 📋 Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [NVIDIA GPU driver](https://www.nvidia.com/download/index.aspx) (for GPU support)
- [Git](https://git-scm.com/downloads)

## 🏃‍♂️ Quick Start

### Windows Users

1. Clone the repository:
   ```bash
   git clone https://github.com/username/deep-learning-tutorials.git
   cd deep-learning-tutorials
   ```

2. Double-click `start.bat` to run
3. Access http://localhost:8888 in your browser

### Linux/Mac Users

```bash
# Clone repository
git clone https://github.com/username/deep-learning-tutorials.git
cd deep-learning-tutorials

# Build and run Docker container
docker-compose up -d

# Access http://localhost:8888 in your browser
```

## 📚 Tutorials

1. **Introduction to Neural Networks**
   - Basic neural network structure
   - Activation functions
   - Loss functions
   - Gradient descent
   - Simple classification example

2. **Convolutional Neural Networks (CNN)**
   - Image classification
   - Feature extraction
   - Transfer learning

3. **Recurrent Neural Networks (RNN)**
   - Sequential data processing
   - Time series analysis
   - Natural language processing

## 🛠️ Technical Details

### Docker Configuration

- Base image: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- Python packages:
  - TensorFlow 2.12+
  - PyTorch 2.0+
  - Jupyter
  - NumPy
  - Pandas
  - Matplotlib

### GPU Support

To verify GPU support:
```bash
nvidia-smi
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📮 Contact

- GitHub Issues: [Create an issue](https://github.com/username/deep-learning-tutorials/issues)
- Email: your.email@example.com

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=username/deep-learning-tutorials&type=Date)](https://star-history.com/#username/deep-learning-tutorials&Date) 