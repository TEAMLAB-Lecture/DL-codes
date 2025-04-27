import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import StanfordDogsDataset

def get_dataloaders(root_dir, batch_size=32, num_workers=4):
    """
    Create training and test DataLoaders
    
    Args:
        root_dir (str): Root directory of the dataset
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, test_loader, num_classes)
    """
    # TODO: Define data augmentation and preprocessing transformations
    # 1. Training data transformations (with augmentation)
    #   - Resize
    #   - Random horizontal flip
    #   - Random rotation
    #   - Color adjustment
    #   - Convert to tensor
    #   - Normalize
    train_transform = transforms.Compose([
        # TODO: Implement training data transformations
    ])
    
    # 2. Test data transformations (basic preprocessing only)
    #   - Resize
    #   - Convert to tensor
    #   - Normalize
    test_transform = transforms.Compose([
        # TODO: Implement test data transformations
    ])
    
    # TODO: Create datasets
    # 1. Create training dataset
    train_dataset = StanfordDogsDataset(
        # TODO: Set parameters
    )
    
    # 2. Create test dataset
    test_dataset = StanfordDogsDataset(
        # TODO: Set parameters
    )
    
    # TODO: Create DataLoaders
    # 1. Training DataLoader
    train_loader = DataLoader(
        # TODO: Set parameters
    )
    
    # 2. Test DataLoader
    test_loader = DataLoader(
        # TODO: Set parameters
    )
    
    # TODO: Get number of classes
    num_classes = # TODO: Set number of classes
    
    return train_loader, test_loader, num_classes 