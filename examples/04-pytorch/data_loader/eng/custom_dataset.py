import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET

class StanfordDogsDataset(Dataset):
    """
    Custom Dataset class for Stanford Dogs Dataset
    
    Args:
        root_dir (str): Root directory of the dataset
        transform (callable, optional): Transform to be applied on images
        train (bool, optional): True for training data, False for test data
    """
    def __init__(self, root_dir, transform=None, train=True):
        # TODO: Implement dataset initialization code
        # 1. Set paths for images and annotations
        # 2. Create mapping between class names and indices
        # 3. Collect image file paths and labels
        # 4. Split into train/test data
        pass
    
    def __len__(self):
        # TODO: Implement code to return the size of the dataset
        pass
    
    def __getitem__(self, idx):
        # TODO: Implement code to return image and label for the given index
        # 1. Load image
        # 2. Parse bounding box information from annotation file
        # 3. Crop image using bounding box
        # 4. Apply transformations
        # 5. Return image and label
        pass
    
    def _parse_annotation(self, annotation_path):
        # TODO: Implement code to parse bounding box information from XML annotation file
        # 1. Parse XML file
        # 2. Extract bounding box coordinates
        # 3. Return coordinates
        pass 