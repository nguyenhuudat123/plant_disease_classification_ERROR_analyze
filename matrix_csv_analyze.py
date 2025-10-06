from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


# CONFUSION MATRIX GENERATION AND CSV EXPORT
# =============================================================================

def save_confusion_matrix_csv(predictions, targets, class_names, model_name, save_dir):
    """Generate and save confusion matrix to CSV file"""
    
    # Generate confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Create DataFrame with class names as indices and columns
    cm_df = pd.DataFrame(cm, 
                        index=[f"True_{class_name}" for class_name in class_names],
                        columns=[f"Pred_{class_name}" for class_name in class_names])
    
    # Save to CSV
    csv_filename = f"{model_name.lower()}_confusion_matrix.csv"
    csv_path = os.path.join(save_dir, csv_filename)
    cm_df.to_csv(csv_path)
    
    print(f"Confusion matrix saved: {csv_path}")
    
    # Also save summary statistics
    stats_filename = f"{model_name.lower()}_classification_stats.csv"
    stats_path = os.path.join(save_dir, stats_filename)
    
    # Calculate per-class metrics
    class_stats = []
    for i, class_name in enumerate(class_names):
        true_positives = cm[i, i]
        false_positives = cm[:, i].sum() - true_positives
        false_negatives = cm[i, :].sum() - true_positives
        true_negatives = cm.sum() - true_positives - false_positives - false_negatives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_stats.append({
            'Class': class_name,
            'True_Positives': true_positives,
            'False_Positives': false_positives,
            'False_Negatives': false_negatives,
            'True_Negatives': true_negatives,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1_score
        })
    
    stats_df = pd.DataFrame(class_stats)
    stats_df.to_csv(stats_path, index=False)
    
    print(f"Classification statistics saved: {stats_path}")
    
    return csv_path, stats_path

def plot_confusion_matrix(cm, class_names, model_name, save_dir):
    """Plot and save confusion matrix visualization"""
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{model_name.lower()}_confusion_matrix.png"
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix plot saved: {plot_path}")
    return plot_path

# =============================================================================
# UPDATED ERROR ANALYSIS
# =============================================================================

def calculate_error_metrics(predictions, targets, probabilities, class_names):
    """Calculate error metrics including confusion matrix"""
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    probabilities = np.array(probabilities)
    
    # Basic metrics
    accuracy = np.mean(predictions == targets) * 100
    error_rate = 100 - accuracy
    total_errors = np.sum(predictions != targets)
    total_samples = len(predictions)
    
    # Confidence analysis
    confidences = np.max(probabilities, axis=1)
    correct_mask = predictions == targets
    
    avg_conf_correct = np.mean(confidences[correct_mask])
    avg_conf_incorrect = np.mean(confidences[~correct_mask])
    confidence_gap = avg_conf_correct - avg_conf_incorrect
    
    # High confidence errors
    high_conf_errors = np.sum((~correct_mask) & (confidences > 0.8))
    
    # Generate confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Per-class error rates
    class_errors = {}
    for i in range(len(class_names)):
        class_mask = targets == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(predictions[class_mask] == targets[class_mask]) * 100
            class_errors[class_names[i]] = 100 - class_accuracy
        else:
            class_errors[class_names[i]] = 0
    
    # Top 5 worst classes
    worst_classes = sorted(class_errors.items(), key=lambda x: x[1], reverse=True)[:5]
    worst_classes = [(name, float(error_rate)) for name, error_rate in worst_classes]
    
    # Most confused pairs
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    
    confused_pairs = sorted(confused_pairs, key=lambda x: x[2], reverse=True)[:5]
    
    return {
        'accuracy': float(accuracy),
        'error_rate': float(error_rate), 
        'total_errors': int(total_errors),
        'total_samples': int(total_samples),
        'avg_confidence_correct': float(avg_conf_correct),
        'avg_confidence_incorrect': float(avg_conf_incorrect),
        'confidence_gap': float(confidence_gap),
        'high_confidence_errors': int(high_conf_errors),
        'worst_classes': worst_classes,
        'most_confused_pairs': confused_pairs,
        'confusion_matrix': cm.tolist()  # Add confusion matrix to results
    }
