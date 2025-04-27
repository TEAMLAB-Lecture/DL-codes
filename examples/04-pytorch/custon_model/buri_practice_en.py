"""
BURI Model Practice Problems

This file contains practice problems for understanding the basic structure and main features of the BURI model.
The problems are designed to increase in difficulty progressively.

Problem 1: Basic BURI Model Implementation
- Complete the BURIModel class.
- Implement a model that takes input size, hidden size, and output size as parameters.
- Use 3 fully connected layers and apply ReLU activation function.

Problem 2: Model Training Function Implementation
- Complete the train_model function.
- Process data in batches and calculate loss.
- Update model parameters using an optimizer.
- Print loss value every 100th batch.

Problem 3: Main Function Implementation
- Set model parameters in the main function.
- Choose appropriate parameter values for the MNIST dataset.
- Create a model instance and set up loss function and optimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Problem 1: BURIModel Class Implementation
class BURIModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # TODO: Write model initialization code
        pass
        
    def forward(self, x):
        # TODO: Write forward propagation function
        pass

# Problem 2: Model Training Function Implementation
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    # TODO: Write model training code
    pass

# Problem 3: Main Function Implementation
def main():
    # TODO: Set model parameters
    input_size = None  # MNIST image size
    hidden_size = None
    output_size = None  # 0-9 digit classification
    
    # TODO: Create model and set up training
    pass

if __name__ == "__main__":
    main() 