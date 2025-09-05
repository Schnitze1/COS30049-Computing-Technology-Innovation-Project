# Network Traffic Anomaly Detection Pipeline

Machine learning pipeline for multiclass network traffic classification and anomaly detection using multiple supervised and unsupervised models.

## Features

- **Multiple Model Types**: Random Forest, MLP (Neural Network), and K-means Clustering
- **Comprehensive Evaluation**: 
  - Multiclass metrics (8 traffic types): Audio, Background, Bruteforce, DoS, Information Gathering, Mirai, Text, Video
  - Binary classification (Malicious vs Benign traffic)
  - Per-class performance analysis
- **Metrics**: Accuracy, Precision, Recall, F1-Score (Weighted), ROC AUC, Confusion Matrices
- **Clustering Evaluation**: Silhouette Score, Calinski-Harabasz, Davies-Bouldin for K-means
- **Automated Reporting**: CSV exports, comprehensive visualizations, and per-class metrics charts
- **Model Caching**: Trained models are cached for faster subsequent runs
- **REST API**: FastAPI-based inference server for real-time predictions
- **Modular Architecture**: Clean separation of data processing, training, evaluation, and serving

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
### 2. Data Preprocessing
1. Open notebook: `data_preprocessing/data_cleaning.ipynb`
2. Set kernel to the current `.venv` interpreter
3. Run all cells to process raw network traffic data
4. This generates `processed_data.npz` and `feature_metadata.pkl`

### 3. Train and Evaluate Models

```bash
# Train models and generate comprehensive evaluation reports
python main.py
```

### 4. Start API Server (Optional)

```bash
# Start FastAPI server for real-time predictions
python serve.py

# Or using uvicorn directly
uvicorn serve:app --host 127.0.0.1 --port 8000 --reload
```

Access API documentation at: `http://127.0.0.1:8000/docs`

## Project Structure

```
├── main.py                          # Main orchestration script
├── train.py                         # Model training functions
├── serve.py                         # FastAPI inference server
├── config.py                        # Configuration settings
├── data_preprocessing/
│   ├── data_cleaning.ipynb          # Data preprocessing notebook
│   ├── input/                       # Raw data files
│   └── output/                      # Processed data files
│       ├── processed_data.npz       # Preprocessed dataset
│       └── feature_metadata.pkl     # Feature metadata and encoders
├── evaluation/
│   ├── calc_eval_metrics.py         # Comprehensive evaluation metrics
│   └── create_reports.py            # Report generation and visualization
├── utils/
│   ├── model_io.py                  # Model saving/loading utilities
│   └── predict.py                   # Prediction functions
├── cache/
│   └── models/                      # Cached trained models
├── evaluation_reports/              # Generated reports and visualizations
│   ├── multiclass_metrics_summary.csv
│   ├── label_from_type_metrics.csv
│   ├── multiclass_metrics_comparison.png
│   ├── confusion_matrices.png
│   ├── per_class_metrics_kmeans.png
│   ├── per_class_metrics_mlp.png
│   └── per_class_metrics_random_forest.png
└── requirements.txt                 # Python dependencies
```

## Models Included

### Supervised Models
- **Random Forest**: Ensemble method with 200 estimators, balanced class weights
- **MLP (Multi-layer Perceptron)**: Neural network with (100, 100) hidden layers, early stopping

### Unsupervised Models
- **K-means Clustering**: 8 clusters with majority vote mapping for evaluation

### Model Features
- **Class Imbalance Handling**: Balanced class weights for Random Forest
- **Early Stopping**: MLP uses early stopping to prevent overfitting
- **Deterministic Results**: All models use random_state=42 for reproducibility
- **Model Caching**: Trained models are automatically saved and reused

## Output Files

The pipeline generates:

### CSV Reports
1. **multiclass_metrics_summary.csv**: Complete multiclass evaluation metrics for all models
2. **label_from_type_metrics.csv**: Binary classification metrics (Malicious vs Benign)

### Visualizations
1. **multiclass_metrics_comparison.png**: Weighted average metrics comparison across models
2. **confusion_matrices.png**: Confusion matrices for all models
3. **per_class_metrics_*.png**: Individual per-class performance charts for each model

### Key Metrics Included
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Macro, Micro, and Weighted averages
- **ROC AUC**: One-vs-Rest multiclass AUC
- **Per-class Metrics**: Individual class performance analysis
- **Clustering Metrics**: Silhouette, Calinski-Harabasz, Davies-Bouldin scores for K-means

## Data Format

### Input Data
Expected data structure in `processed_data.npz`:
- `X_train`: Training features (standardized)
- `X_test`: Test features (standardized)
- `y_train`: Training labels (encoded)
- `y_test`: Test labels (encoded)

### Traffic Types (8 classes)
- **Benign**: Audio, Background, Text, Video
- **Malicious**: Bruteforce, DoS, Information Gathering, Mirai

### Feature Metadata
`feature_metadata.pkl` contains:
- Label encoder for traffic types
- Feature names and preprocessing information
- Target variable configuration

## API Usage

### Start the Server
```bash
python serve.py
```

### Make Predictions
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "random_forest",
       "instances": [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
     }'
```

### Available Endpoints
- `GET /api/v1/health`: Health check
- `GET /api/v1/models`: List available models
- `POST /api/v1/predict`: Make predictions
- `GET /docs`: Interactive API documentation

## Requirements

- Python 3.8+
- scikit-learn
- fastapi
- uvicorn
- pandas
- matplotlib
- seaborn
- numpy
- joblib

See `requirements.txt` for specific versions.

## Performance Results

Based on the latest evaluation:

| Model | Accuracy | F1-Score (Weighted) | Best For |
|-------|----------|-------------------|----------|
| Random Forest | 97.44% | 97.51% | Overall best performance |
| MLP | 90.01% | 91.03% | Good balance of performance |
| K-means | 62.14% | 55.34% | Unsupervised baseline |

**Note**: Information Gathering class has only 1 test sample, affecting per-class metrics for all models.
