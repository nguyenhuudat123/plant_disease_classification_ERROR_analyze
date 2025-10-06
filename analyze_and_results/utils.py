"""
Shared utilities for plant disease classification analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r'F:\up_git\off_plant_di_err_analyze_daok\analyze_and_results')
PHOTO_DIR = BASE_DIR / 'photos'
CSV_DIR = BASE_DIR / 'reports'

# Create directories
PHOTO_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data(models_path='models_info.csv',
                  class_stats_path='combined_classification_stats_wide.csv',
                  confusion_path='combined_confusion_matrix_wide.csv'):
    """Load all datasets"""
    df_models = pd.read_csv(models_path)
    df_class_stats = pd.read_csv(class_stats_path)
    df_confusion = pd.read_csv(confusion_path)
    return df_models, df_class_stats, df_confusion

def get_pred_columns(df_confusion):
    """Extract prediction columns from confusion matrix"""
    pred_cols = [col for col in df_confusion.columns if col.startswith('Pred_')]
    all_classes = sorted([col.replace('Pred_', '') for col in pred_cols])
    return pred_cols, all_classes

# ============================================================================
# FILE SAVING
# ============================================================================

def setup_plotting_style():
    """Set up consistent plotting style"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

def save_figure(filename, dpi=300):
    """Save figure to photos directory"""
    filepath = PHOTO_DIR / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure: {filepath}")

def save_csv(df, filename):
    """Save dataframe to CSV in reports directory"""
    filepath = CSV_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"Saved CSV: {filepath}")

# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def calculate_efficiency_score(df_models):
    """Calculate efficiency score for models"""
    return (
        df_models['Final_Accuracy'] / 
        (df_models['Training_Time_Minutes'] * df_models['Error_Rate'] * 
         df_models['Parameters_Million'])
    ) * 1000

def calculate_research_score(df_models):
    """Calculate score for research use case"""
    return (
        df_models['Final_Accuracy'] * 0.7 +
        (100 - df_models['Error_Rate']) * 0.2 +
        (df_models['Confidence_Gap'] * 100) * 0.1
    )

def calculate_production_score(df_models):
    """Calculate score for production use case"""
    return (
        df_models['Final_Accuracy'] * 0.4 +
        (100 - df_models['Error_Rate']) * 0.2 +
        (df_models['Confidence_Gap'] * 100) * 0.2 +
        (100 / (1 + df_models['Training_Time_Minutes'])) * 0.1 +
        (100 - df_models['High_Confidence_Errors'] / 
         df_models['Total_Errors'] * 100) * 0.1
    )

def calculate_speed_score(df_models):
    """Calculate score for speed-critical use case"""
    return (
        (100 / (1 + df_models['Training_Time_Minutes'])) * 0.5 +
        (100 / (1 + df_models['Parameters_Million'])) * 0.3 +
        df_models['Final_Accuracy'] * 0.2
    )

def calculate_reliability_score(df_models):
    """Calculate score for reliability-critical use case"""
    return (
        (df_models['Confidence_Gap'] * 100) * 0.4 +
        (100 - df_models['High_Confidence_Errors'] / 
         df_models['Total_Errors'] * 100) * 0.3 +
        (100 - df_models['Accuracy_Drop']) * 0.2 +
        df_models['Final_Accuracy'] * 0.1
    )

# ============================================================================
# CONFIDENCE ANALYSIS
# ============================================================================

def calculate_overconfidence_rate(df_models):
    """Calculate overconfidence rate"""
    return (df_models['High_Confidence_Errors'] / 
            df_models['Total_Errors'] * 100)

def calculate_risk_score(df_models):
    """Calculate risk score"""
    return (df_models['High_Confidence_Errors'] * 
            df_models['Avg_Confidence_Incorrect'])

def calculate_confidence_threshold(row):
    """Calculate recommended confidence threshold"""
    return row['Avg_Confidence_Incorrect'] + (row['Confidence_Gap'] * 0.5)

def categorize_model_by_confidence(row, high_conf_gap_threshold, 
                                   high_overconf_threshold):
    """Categorize model based on confidence behavior"""
    if (row['Confidence_Gap'] >= high_conf_gap_threshold and 
        row['Overconfidence_Rate'] < high_overconf_threshold):
        return "Well-Calibrated & Safe"
    elif (row['Confidence_Gap'] >= high_conf_gap_threshold and 
          row['Overconfidence_Rate'] >= high_overconf_threshold):
        return "Well-Calibrated but Risky"
    elif (row['Confidence_Gap'] < high_conf_gap_threshold and 
          row['Overconfidence_Rate'] < high_overconf_threshold):
        return "Poorly-Calibrated but Cautious"
    else:
        return "Poorly-Calibrated & Dangerous"

# ============================================================================
# ERROR ANALYSIS
# ============================================================================

def build_error_frequency_df(df_confusion, pred_cols):
    """Build error frequency dataframe"""
    error_frequency = []
    
    for idx, row in df_confusion.iterrows():
        true_class = row['Unnamed: 0']
        model = row['model_name']
        
        for pred_col in pred_cols:
            pred_class = pred_col.replace('Pred_', '')
            count = row[pred_col]
            
            if true_class != pred_class and count > 0:
                error_frequency.append({
                    'True_Class': true_class,
                    'Predicted_Class': pred_class,
                    'Model': model,
                    'Count': count
                })
    
    return pd.DataFrame(error_frequency)

def categorize_errors_by_frequency(error_model_count, total_models):
    """Categorize errors as hard, moderate, or soft"""
    hard_errors = error_model_count[
        error_model_count['Num_Models'] == total_models
    ].copy()
    
    moderate_errors = error_model_count[
        (error_model_count['Num_Models'] >= total_models * 0.75) &
        (error_model_count['Num_Models'] < total_models)
    ].copy()
    
    soft_errors = error_model_count[
        error_model_count['Num_Models'] < total_models * 0.75
    ].copy()
    
    return hard_errors, moderate_errors, soft_errors

# ============================================================================
# CONFUSION MATRIX UTILITIES
# ============================================================================

def aggregate_confusion_matrix(df_confusion, pred_cols, top_n=30):
    """Calculate aggregated confusion matrix across all models"""
    avg_confusion = df_confusion.groupby('Unnamed: 0')[pred_cols].mean()
    avg_confusion.columns = [col.replace('Pred_', '') for col in avg_confusion.columns]
    
    if len(avg_confusion) > top_n:
        row_sums = avg_confusion.sum(axis=1)
        col_sums = avg_confusion.sum(axis=0)
        all_confusion_scores = pd.concat([row_sums, col_sums]).groupby(level=0).sum()
        top_classes = all_confusion_scores.nlargest(top_n).index.tolist()
        
        valid_classes = [c for c in top_classes 
                        if c in avg_confusion.index and c in avg_confusion.columns]
        
        if valid_classes:
            avg_confusion = avg_confusion.loc[valid_classes, valid_classes]
    
    return avg_confusion

# ============================================================================
# CLASS PERFORMANCE ANALYSIS
# ============================================================================

def analyze_class_performance(df_class_stats):
    """Aggregate class performance metrics across models"""
    class_performance = df_class_stats.groupby('Class').agg({
        'F1_Score': ['mean', 'std', 'min', 'max'],
        'Precision': ['mean', 'min'],
        'Recall': ['mean', 'min'],
        'True_Positives': 'sum',
        'False_Negatives': 'sum',
        'False_Positives': 'sum'
    }).reset_index()
    
    class_performance.columns = [
        'Class', 'F1_Mean', 'F1_Std', 'F1_Min', 'F1_Max',
        'Precision_Mean', 'Precision_Min', 'Recall_Mean', 'Recall_Min',
        'TP_Sum', 'FN_Sum', 'FP_Sum'
    ]
    
    class_performance['Total_Samples'] = class_performance['TP_Sum'] + class_performance['FN_Sum']
    class_performance['Error_Rate'] = (
        (class_performance['FN_Sum'] + class_performance['FP_Sum']) / 
        class_performance['Total_Samples'] * 100
    )
    
    return class_performance

def identify_problematic_classes(class_performance, f1_threshold=None, 
                                 variance_threshold=None):
    """Identify classes needing attention"""
    if f1_threshold is None:
        f1_threshold = class_performance['F1_Mean'].median()
    if variance_threshold is None:
        variance_threshold = class_performance['F1_Std'].median()
    
    needs_more_data = class_performance[
        (class_performance['F1_Mean'] < f1_threshold) &
        (class_performance['F1_Std'] > variance_threshold)
    ]
    
    return needs_more_data.sort_values('F1_Mean')

# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def create_horizontal_bar(ax, data, labels, title, xlabel, colors=None, 
                         invert_y=True):
    """Create standardized horizontal bar chart"""
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data)))
    
    ax.barh(range(len(data)), data, color=colors, alpha=0.7, 
            edgecolor='black', linewidth=1.5)  # Changed from edgecolors to edgecolor
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    if invert_y:
        ax.invert_yaxis()
    return ax

def truncate_label(label, max_length=30):
    """Truncate long labels"""
    return label[:max_length] if len(label) > max_length else label

# ============================================================================
# PRINTING UTILITIES
# ============================================================================

def print_section_header(title, char="=", width=80):
    """Print formatted section header"""
    print("\n" + char*width)
    print(title)
    print(char*width)

def print_subsection_header(title, char="-", width=80):
    """Print formatted subsection header"""
    print("\n" + char*width)
    print(title)
    print(char*width)

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def calculate_correlations(df_models, metrics):
    """Calculate correlation matrix for specified metrics"""
    return df_models[metrics].corr()

def print_correlation(col1, col2, df_models):
    """Calculate and print Pearson correlation"""
    corr, p_value = stats.pearsonr(df_models[col1], df_models[col2])
    print(f"Correlation: {col1} <-> {col2}: {corr:.4f} (p={p_value:.4f})")
    return corr, p_value