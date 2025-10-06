# DATA LOADING
# =============================================================================
import torch
from torch.utils.data import DataLoader, Subset
import random
import pickle


def load_data(path, subset_ratio=0.1):
    """Load data with optional subset percentage"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    if subset_ratio >= 1.0:
        return data['train_loader'], data['val_loader'], data['classes']
    
    train_dataset = data['train_loader'].dataset
    val_dataset = data['val_loader'].dataset
    
    train_size = int(len(train_dataset) * subset_ratio)
    val_size = int(len(val_dataset) * subset_ratio)
    
    print(f"Using {subset_ratio*100:.1f}% of data:")
    print(f"  Train: {train_size}/{len(train_dataset)} samples")
    print(f"  Val: {val_size}/{len(val_dataset)} samples")
    
    random.seed(42)
    train_indices = random.sample(range(len(train_dataset)), train_size)
    val_indices = random.sample(range(len(val_dataset)), val_size)
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, data['classes']