# Stanford Dogs Dataset Dog Breed Classification Problem

## Problem Description
Implement a deep learning model to classify dog breeds using the Stanford Dogs Dataset. 
This dataset consists of 120 different dog breeds, with each image containing a dog of the corresponding breed.

## Implementation Requirements

### 1. Data Preprocessing
- Create `custom_dataset.py` file and implement the `StanfordDogsDataset` class.
  - Must inherit from `torch.utils.data.Dataset`.
  - Must have functionality to load images and labels.
  - Must crop only the dog part using bounding box information.
- Create `custom_dataloader.py` file and implement the data loader.
  - Must separate training and test datasets.
  - Must apply data augmentation techniques.
  - Must be able to load data in batches.

### 2. Model Implementation
- Create `custom_model.py` file and implement the model.
  - Use ResNet18 as the base model.
  - Modify the last fully connected layer to classify 120 classes.
  - Freeze all layers except the last layer.

### 3. Training and Evaluation
- Create `train.py` file and implement the training code.
  - Use Adam optimizer.
  - Use Cross Entropy Loss as the loss function.
  - Set learning rate to 0.001.
  - Train for 10 epochs.
  - Print training and validation accuracy for each epoch.
  - Visualize the training process.

## References
1. PyTorch Official Documentation: https://pytorch.org/docs/stable/index.html
2. torchvision Models: https://pytorch.org/vision/stable/models.html
3. Stanford Dogs Dataset: http://vision.stanford.edu/aditya86/ImageNetDogs/

## Evaluation Criteria
- Accuracy of model implementation
- Code readability and structure
- Training process stability
- Final model performance

## Submission
1. Implement and submit the following files:
   - `custom_dataset.py`
   - `custom_dataloader.py`
   - `custom_model.py`
   - `train.py`
2. Training result graphs
3. Final model accuracy
4. Difficulties or notable points during implementation

## Hints
1. When implementing the Dataset class, the following methods are needed:
   - `__init__`: Initialize the dataset
   - `__len__`: Return the size of the dataset
   - `__getitem__`: Return data for the given index
2. When implementing the DataLoader, consider the following parameters:
   - `batch_size`
   - `shuffle`
   - `num_workers`
3. When implementing the model, consider:
   - Loading pre-trained model
   - Modifying layers
   - Freezing parameters
4. When implementing the training code, consider:
   - Optimizer settings
   - Loss function settings
   - Training loop implementation
   - Performance evaluation 