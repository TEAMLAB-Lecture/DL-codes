import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tqdm import tqdm

class WandbMonitor:
    def __init__(self, project_name, config):
        """
        Initialize W&B monitoring
        
        Args:
            project_name (str): W&B project name
            config (dict): Configuration dictionary
        """
        # Initialize W&B
        wandb.init(
            project=project_name,
            config=config,
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Store configuration
        self.config = wandb.config
        
    def watch_model(self, model, log_freq=100):
        """
        Watch model for gradient logging
        
        Args:
            model (nn.Module): PyTorch model
            log_freq (int): Logging frequency
        """
        wandb.watch(model, log_freq=log_freq)
        
    def log_metrics(self, metrics, step=None):
        """
        Log metrics to W&B
        
        Args:
            metrics (dict): Dictionary of metrics to log
            step (int, optional): Step number
        """
        wandb.log(metrics, step=step)
        
    def log_images(self, images, name="images", captions=None):
        """
        Log images to W&B
        
        Args:
            images (torch.Tensor): Batch of images
            name (str): Name for the image group
            captions (list, optional): List of captions for each image
        """
        # Convert tensor to numpy and normalize
        images_np = images.cpu().numpy()
        images_np = (images_np - images_np.min()) / (images_np.max() - images_np.min())
        
        # Create wandb images
        wandb_images = []
        for i, img in enumerate(images_np):
            if captions and i < len(captions):
                wandb_images.append(wandb.Image(img, caption=captions[i]))
            else:
                wandb_images.append(wandb.Image(img))
                
        wandb.log({name: wandb_images})
        
    def log_confusion_matrix(self, y_true, y_pred, class_names):
        """
        Log confusion matrix to W&B
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (list): List of class names
        """
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )
        })
        
    def log_learning_rate(self, optimizer, step):
        """
        Log learning rate to W&B
        
        Args:
            optimizer (optim.Optimizer): PyTorch optimizer
            step (int): Step number
        """
        lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": lr}, step=step)
        
    def log_gradients(self, model, step):
        """
        Log gradient statistics to W&B
        
        Args:
            model (nn.Module): PyTorch model
            step (int): Step number
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                wandb.log({
                    f"gradients/{name}/mean": param.grad.mean().item(),
                    f"gradients/{name}/std": param.grad.std().item(),
                    f"gradients/{name}/max": param.grad.max().item(),
                    f"gradients/{name}/min": param.grad.min().item()
                }, step=step)
                
    def log_model_weights(self, model, step):
        """
        Log model weight statistics to W&B
        
        Args:
            model (nn.Module): PyTorch model
            step (int): Step number
        """
        for name, param in model.named_parameters():
            wandb.log({
                f"weights/{name}/mean": param.data.mean().item(),
                f"weights/{name}/std": param.data.std().item(),
                f"weights/{name}/max": param.data.max().item(),
                f"weights/{name}/min": param.data.min().item()
            }, step=step)
            
    def log_artifact(self, name, type, description, path):
        """
        Log artifact to W&B
        
        Args:
            name (str): Artifact name
            type (str): Artifact type
            description (str): Artifact description
            path (str): Path to the artifact
        """
        artifact = wandb.Artifact(name, type=type, description=description)
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        
    def finish(self):
        """Finish W&B run"""
        wandb.finish()

def train_model(model, train_loader, val_loader, criterion, optimizer, monitor, num_epochs, device):
    """
    Train model with W&B monitoring
    
    Args:
        model (nn.Module): PyTorch model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        monitor (WandbMonitor): W&B monitor instance
        num_epochs (int): Number of epochs
        device (torch.device): Device to use
    """
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': train_loss / (train_bar.n + 1),
                'acc': 100. * train_correct / train_total
            })
            
            # Log batch metrics
            if train_bar.n % 100 == 0:
                monitor.log_metrics({
                    'batch_loss': loss.item(),
                    'batch_acc': 100. * predicted.eq(labels).sum().item() / labels.size(0)
                }, step=epoch * len(train_loader) + train_bar.n)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Store predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                val_bar.set_postfix({
                    'loss': val_loss / (val_bar.n + 1),
                    'acc': 100. * val_correct / val_total
                })
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Log epoch metrics
        monitor.log_metrics({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }, step=epoch)
        
        # Log learning rate
        monitor.log_learning_rate(optimizer, epoch)
        
        # Log gradients and weights
        monitor.log_gradients(model, epoch)
        monitor.log_model_weights(model, epoch)
        
        # Log example images
        if epoch % 5 == 0:  # Log images every 5 epochs
            example_images = inputs[:4]  # First 4 images from last batch
            monitor.log_images(example_images, "example_images")
        
        # Log confusion matrix
        if epoch % 10 == 0:  # Log confusion matrix every 10 epochs
            monitor.log_confusion_matrix(
                np.array(all_labels),
                np.array(all_preds),
                [str(i) for i in range(120)]  # Assuming 120 classes
            )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            monitor.log_artifact(
                'best_model',
                'model',
                f'Best model at epoch {epoch+1} with val_acc {val_acc:.2f}',
                'best_model.pth'
            )

def main():
    # Configuration
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "model": "ResNet18",
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # Initialize monitor
    monitor = WandbMonitor("dog_breed_classification", config)
    
    # Set device
    device = torch.device(config["device"])
    
    # Load and prepare data
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize model
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 120)  # 120 dog breeds
    
    # Freeze all layers except the last one
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    model = model.to(device)
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.fc.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # Watch model
    monitor.watch_model(model)
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,  # Replace with actual train_loader
        val_loader=val_loader,      # Replace with actual val_loader
        criterion=criterion,
        optimizer=optimizer,
        monitor=monitor,
        num_epochs=config["epochs"],
        device=device
    )
    
    # Finish monitoring
    monitor.finish()

if __name__ == "__main__":
    main()
