from typing import List, Dict, Tuple
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
from sklearn.model_selection import train_test_split
import numpy as np

data_path = 'dataset/'


class TransformDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that applies transforms
    Defined at module level for Windows pickle compatibility
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)


def prepare_data():
    """
    Optimized data preparation for 97-99% accuracy
    Windows-compatible version (no multiprocessing issues)
    """
    
    # Batch size optimized for 128x128 images
    BATCH_SIZE = 24
    random_state = 42
    
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)
    
    # TRAINING TRANSFORMS - Moderate augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((140, 140)),
        transforms.RandomCrop(128),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # VALIDATION TRANSFORMS - No augmentation
    val_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = torchvision.datasets.ImageFolder(data_path)
    classes = full_dataset.classes
    targets = full_dataset.targets
    
    # Print dataset info
    print(f"\n{'='*60}")
    print(f"DATASET INFORMATION")
    print(f"{'='*60}")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Classes: {classes}")
    
    # Check class balance
    unique, counts = np.unique(targets, return_counts=True)
    for cls, count in zip(classes, counts):
        print(f"  {cls}: {count} images ({count/len(full_dataset)*100:.1f}%)")
    
    # Calculate class weights for balanced training
    class_weights = 1. / torch.tensor(counts, dtype=torch.float)
    sample_weights = class_weights[targets]
    
    # Split dataset (70-30 split)
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=0.3,
        stratify=targets,
        random_state=random_state
    )
    
    print(f"\nTraining samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"{'='*60}\n")
    
    # Get original images without transform
    original_dataset = torchvision.datasets.ImageFolder(data_path, transform=None)
    train_subset = torch.utils.data.Subset(original_dataset, train_indices)
    val_subset = torch.utils.data.Subset(original_dataset, val_indices)
    
    # Apply transforms using the module-level class
    train_dataset = TransformDataset(train_subset, train_transforms)
    val_dataset = TransformDataset(val_subset, val_transforms)
    
    # Weighted sampler for balanced training
    train_sample_weights = sample_weights[train_indices]
    sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_sample_weights),
        replacement=True
    )
    
    # Create loaders - num_workers=0 for Windows compatibility
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,  # Set to 0 for Windows - no multiprocessing
        pin_memory=False  # Disabled for CPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows - no multiprocessing
        pin_memory=False  # Disabled for CPU
    )
    
    return train_loader, val_loader, classes