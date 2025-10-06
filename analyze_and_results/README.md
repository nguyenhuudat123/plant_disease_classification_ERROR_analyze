# Plant Disease Classification: Multi-Model Error Analysis

A comprehensive benchmarking and error analysis framework for comparing 12 deep learning architectures on plant disease detection across 38 disease classes.

## 📊 Project Overview

This project provides an end-to-end pipeline for:
- Training multiple CNN architectures on plant disease classification
- Comprehensive error pattern analysis
- Confidence calibration assessment
- Model comparison and deployment recommendations
- Publication-ready visualizations

### Key Features

- **12 Model Architectures**: SimpleCNN, LeNet, AlexNet, ResNet (18/34/50), DenseNet121, MobileNet (V2/V3), ShuffleNet, EfficientNet, InceptionV3
- **38 Disease Classes**: Covering Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato
- **Advanced Analysis**: Error patterns, confusion matrices, confidence calibration, systematic errors, model specialization
- **Publication-Ready**: 7 figures + comprehensive tables for research papers

## 🗂️ Project Structure

```
.
├── main.ipynb              # Main training pipeline
├── model_lib.py                    # Model architectures
├── train_analyze_lib.py            # Training and analysis functions
├── load_data_lib.py                # Data loading utilities
├── matrix_csv_analyze.py           # Confusion matrix analysis
├── error_visualize.py              # Error visualization tools
├── plant_data_tvt.pkl              # Dataset (train/val/test splits)
│
├── models/                         # Trained model weights (.pth files)
│   ├── resnet50_final.pth
│   ├── efficientnetb0_final.pth
│   └── ...
│
├── results/                        # Per-model results
│   ├── {model}_confusion_matrix.csv
│   ├── {model}_classification_stats.csv
│   ├── {model}_confusion_matrix.png
│   └── misclassified_*.png
│
└── analyze_and_results/            # Comprehensive analysis
    ├── utils.py                    # Shared utilities
    ├── summary_analyze.ipynb       # All analysis parts
    ├── models_info.csv             # Model comparison metrics
    ├── combined_classification_stats_wide.csv
    ├── combined_confusion_matrix_wide.csv
    │
    ├── photos/                     # Generated visualizations
    │   ├── paper_fig1_aggregated_confusion.png
    │   ├── paper_fig2_scatter_matrix.png
    │   ├── paper_fig3_radar_chart.png
    │   └── ...
    │
    └── reports/                    # Analysis results (CSV)
        ├── step2_top_confusion_pairs.csv
        ├── step3_deployment_recommendations.csv
        └── ...
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm
```

### 1. Training Models

```python
# Run the main training notebook
jupyter notebook main_ngangon.ipynb

# Or run directly
python main_ngangon.py  # (if converted from notebook)
```

**Configuration:**
- `DATA_SUBSET_RATIO`: 1.0 (use 100% of data)
- `EPOCHS`: 40
- Mixed precision training enabled
- AdamW optimizer with CosineAnnealingLR

### 2. Analysis Pipeline

Navigate to `analyze_and_results/` and run:

```python
# Part 1: Model Overview & Rankings
jupyter notebook 01_model_overview.ipynb

# Part 2: Error Pattern Analysis
jupyter notebook 02_error_patterns.ipynb

# Part 3: Confidence Analysis
jupyter notebook 03_confidence_analysis.ipynb

# Part 4: Error Type Deep Dive
jupyter notebook 04_error_deep_dive.ipynb

# Or run all at once
jupyter notebook summary_analyze.ipynb
```

## 📈 Analysis Components

### Part 1: Model Overview & Rankings
- Top models by accuracy, speed, efficiency, and size
- Trade-off analysis (accuracy vs time, size vs performance)
- Correlation matrix of performance metrics
- Efficiency score calculation

**Output:** `step1_overview_analysis.png`, `step1_correlation_matrix.png`

### Part 2: Error Pattern Analysis
- Top 20 most confused class pairs
- Systematic errors (made by all models)
- Problematic classes identification
- Per-class F1-score distribution
- High variance classes (ensemble candidates)

**Output:** `step2_error_analysis.png`, multiple CSVs

### Part 3: Confidence Calibration
- Confidence gap analysis
- Overconfidence rate calculation
- Risk score assessment
- Model categorization (4 categories)
- Deployment recommendations

**Output:** `step3_confidence_analysis.png`, deployment recommendations

### Part 4: Error Type Deep Dive
- Hard errors (100% models fail)
- Soft errors (model specialization)
- High-confidence errors (dangerous predictions)
- Confusion clusters (hierarchical)

**Output:** `step4_error_types.png`, detailed error CSVs

### Part 6: Paper-Ready Visualizations
- Aggregated confusion matrix
- Scatter plot matrix
- Radar chart (top 5 models)
- Pareto frontier
- Per-model confusion grid
- Box plots (class variation)
- Comprehensive overview dashboard

**Output:** 7 publication-ready figures + summary table

## 📊 Key Results

### Model Performance Summary

| Model | Accuracy (%) | Training Time (min) | Parameters (M) | Confidence Gap |
|-------|-------------|---------------------|----------------|----------------|
| ResNet50 | XX.XX | XX.X | XX.X | X.XXX |
| EfficientNetB0 | XX.XX | XX.X | XX.X | X.XXX |
| ... | ... | ... | ... | ... |

*Run analysis to generate actual results*

### Key Findings
- **Best Overall**: [To be determined after analysis]
- **Most Efficient**: [Model with best accuracy/time ratio]
- **Best Calibrated**: [Model with highest confidence gap]
- **Deployment Ready**: [Top recommendation for production]

### Systematic Errors
- Identified XX hard errors (100% models fail)
- Top confused pairs: [Class A → Class B]
- Most problematic class: [Class name]

## 🔬 Methodology

### Training Configuration
- **Optimizer**: AdamW (lr=0.001, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR
- **Loss**: CrossEntropyLoss
- **Mixed Precision**: Enabled (AMP)
- **Batch Size**: 64
- **Epochs**: 40

### Metrics Collected
- Accuracy (Top-1, Top-3, Top-5)
- Per-class Precision, Recall, F1-Score
- Confusion Matrix (38×38)
- Confidence scores (correct vs incorrect)
- High-confidence errors
- Training time and model parameters

### Analysis Framework
- **Efficiency Score**: `Accuracy / (Time × Error_Rate × Parameters)`
- **Risk Score**: `High_Conf_Errors × Avg_Conf_Incorrect`
- **Deployment Score**: Weighted combination for medical use
- **Error Categorization**: Hard (100%), Moderate (75-99%), Soft (<75%)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{plant_disease_multimodel_analysis,
  author = {Your Name},
  title = {Plant Disease Classification: Multi-Model Error Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/nguyenhuudat123/plant_disease_classification_ERROR_analyze}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaboration:
- GitHub: https://github.com/nguyenhuudat123

## 🙏 Acknowledgments

- Plant Village Dataset
- PyTorch Team
- Open-source community

## 🐛 Known Issues

- Large file sizes for trained models (use Git LFS)
- High memory usage during confusion matrix visualization
- Requires CUDA-capable GPU for optimal training speed

## 🔮 Future Work

- [ ] Add more model architectures (Vision Transformers, ConvNeXt)
- [ ] Implement ensemble learning based on soft error analysis
- [ ] Add explainability methods (GradCAM, LIME)
- [ ] Deploy as REST API for real-time inference
- [ ] Mobile app integration
- [ ] Multi-language support for class names

---

**Last Updated**: January 2025  
**Status**: Active Development