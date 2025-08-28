# Classification Model Evaluation Pipeline

Python pipeline for training and evaluating multiple classification models.

## Features

- **Current Models**: Random Forest, Logistic Regression, SVM, MLP, and Deep Neural Network
- **Evaluation Metrics**: Accuracy, F1-score, Precision, Recall, ROC AUC, Confusion Matrix
- **Automated Reporting**: CSV exports and visualization charts
- **Modularised Functions**: Data loading, training, evaluation, and reporting

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### 2. Run Jupyter notebook for Data processing
1. Open notebook in: data_prerprocessing\data_cleaning.ipynb
2. Set kernel with the current .venv interpreter
3. Run notebook

### 3. Run Evaluation

```bash
# Basic run (uses default data paths)
python main.py

# With custom data paths
python main.py --data path/to/data.npz --features path/to/metadata.pkl
```

## Project Structure

```
├── main.py                          # Main orchestration script
├── train.py                         # Model training functions
├── data_prerprocessing/
│   └── load_data.py                 # Data loading utilities
├── evaluation/
│   ├── calc_eval_metrics.py         # Evaluation metrics calculation
│   └── create_reports.py            # Report generation and visualization
├── models/
│   └── deep_nn.py                   # Deep learning model wrapper
├── requirements.txt                 # Python dependencies
└── evaluation_reports/              # Generated reports and charts
    ├── metrics.csv                  # Evaluation metrics
    ├── all_metrics_bar_charts.png   # Combined metrics visualization
    └── confusion_matrices_grid.png  # Confusion matrices heatmap
```

## Models Included

- **Random Forest**: Ensemble method with 200 estimators
- **Logistic Regression**: Linear classifier with L2 regularization
- **SVM (RBF)**: Support Vector Machine with RBF kernel
- **MLP**: Multi-layer Perceptron (64, 32 hidden layers)
- **Deep NN**: Keras-based neural network (128, 64, 32 hidden layers)

## Output Files

The pipeline generates:

1. **metrics.csv**: Tabular results with all evaluation metrics
2. **all_metrics_bar_charts.png**: Combined bar chart of F1, ROC AUC, Accuracy, Precision, Recall
3. **confusion_matrices_grid.png**: Heatmap grid of all models' confusion matrices

## Data Format

Expected data structure in `.npz` file:
- `X_train`: Training features
- `X_test`: Test features  
- `y_train`: Training labels
- `y_test`: Test labels

## Requirements

- Python 3.8+
- scikit-learn
- tensorflow
- pandas
- matplotlib
- seaborn
- numpy

See `requirements.txt` for specific versions.
