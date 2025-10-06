import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import json
import time
import pickle
import os
import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms

from matrix_csv_analyze import *
from error_visualize import save_misclassified_images, compute_topk_accuracy


# =============================================================================
# DEVICE
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in tqdm(loader, desc="Training"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, pred = torch.max(output, 1)
        total += target.size(0)
        correct += (pred == target).sum().item()

    return total_loss / len(loader), 100. * correct / total


def validate_with_details(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in tqdm(loader, desc="Validation"):
            data, target = data.to(device), target.to(device)

            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)

            total_loss += loss.item()
            probs = torch.softmax(output, dim=1)
            _, pred = torch.max(probs, 1)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = 100. * sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_preds)
    return total_loss / len(loader), accuracy, all_preds, all_targets, all_probs


# =============================================================================
# TRAIN & ANALYZE
# =============================================================================
def train_and_analyze(model, model_name, train_loader, val_loader, classes, save_dir, epochs=15, save_miscls=True):
    """Train model and analyze errors with confusion matrix export + extra visualizations"""

    print(f"\nTraining {model_name}...")

    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0
    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, val_preds, val_targets, val_probs = validate_with_details(model, val_loader, criterion)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc

        print(f"Epoch {epoch+1}: Train {train_acc:.1f}%, Val {val_acc:.1f}%")

    training_time = time.time() - start_time

    # Final validation for error analysis
    _, final_acc, final_preds, final_targets, final_probs = validate_with_details(model, val_loader, criterion)

    # Calculate error metrics (includes confusion matrix)
    error_metrics = calculate_error_metrics(final_preds, final_targets, final_probs, classes)

    # Save confusion matrix to CSV and plot
    cm_csv_path, stats_csv_path = save_confusion_matrix_csv(
        final_preds, final_targets, classes, model_name, save_dir
    )

    cm_plot_path = plot_confusion_matrix(
        np.array(error_metrics['confusion_matrix']), classes, model_name, save_dir
    )

    # Compute Top-k accuracy
    topk_results = compute_topk_accuracy(final_probs, final_targets, ks=(1, 3, 5))
    print("Top-k Accuracy:", topk_results)

    # Save misclassified images
    if save_miscls:
        val_dataset = val_loader.dataset
        save_misclassified_images(val_dataset, final_preds, final_targets, final_probs, classes, save_dir, n=20)

    # Print summary
    print(f"\n{model_name} RESULTS:")
    print("-" * 40)
    print(f"Accuracy: {error_metrics['accuracy']:.2f}%")
    print(f"Error rate: {error_metrics['error_rate']:.2f}%")
    print(f"Files saved:")
    print(f"  - Confusion Matrix CSV: {cm_csv_path}")
    print(f"  - Classification Stats: {stats_csv_path}")
    print(f"  - Confusion Matrix Plot: {cm_plot_path}")

    # Prepare results
    results = {
        'model_name': model_name,
        'parameters': int(sum(p.numel() for p in model.parameters())),
        'best_accuracy': float(best_acc),
        'final_accuracy': float(final_acc),
        'training_time_minutes': float(training_time / 60),
        'error_metrics': error_metrics,
        'confusion_matrix_csv': cm_csv_path,
        'classification_stats_csv': stats_csv_path,
        'confusion_matrix_plot': cm_plot_path,
        'topk_accuracy': topk_results
    }

    return results, val_loader
