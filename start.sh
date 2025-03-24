#!/bin/bash

echo "Deep Learning Course Docker Environment Starting..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running."
    echo "Please start Docker and try again."
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU not detected. Running without GPU support..."
else
    echo "NVIDIA GPU detected. Running with GPU support..."
fi

# Start Docker container
echo "Starting Docker container..."
cd examples
docker-compose up --build

echo
echo "Jupyter Notebook has started."
echo "Please access http://localhost:8888 in your browser."
echo
echo "Press Ctrl+C to exit..."

# Stop container
docker-compose down
cd .. 