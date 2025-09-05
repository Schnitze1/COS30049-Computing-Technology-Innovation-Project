"""
Multiclass Network Traffic Classification Evaluation
Evaluates models for multiclass threat type classification
"""

from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_fscore_support
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def kmeans_eval(km_model, X, y_true):
    """
    Evaluate K-means clustering for multiclass using majority vote
    Uses the already trained K-means model
    """
    clusters = km_model.predict(X)

    # Majority vote: for each cluster, pick the most frequent y_true
    mapping = {}
    n_clusters = km_model.n_clusters
    for c in range(n_clusters):
        idx = np.where(clusters == c)[0]
        if len(idx) == 0:
            mapping[c] = 0  # default if empty
            continue
        # pick the mode label inside cluster c
        vals, counts = np.unique(y_true[idx], return_counts=True)
        mapping[c] = int(vals[np.argmax(counts)])

    # convert clusters -> predicted labels
    y_pred = np.array([mapping[c] for c in clusters], dtype=int)

    return y_pred, mapping

def evaluate_models(models: Dict[str, object], X_test, y_test, n_classes: int) -> Dict[str, Dict[str, object]]:
    """
    Evaluate multiclass models and return comprehensive metrics
    """
    results: Dict[str, Dict[str, object]] = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        if name == 'kmeans':  # Unsupervised: clustering with multiclass evaluation
            y_pred = model.predict(X_test)
            
            # Traditional clustering metrics
            clustering_metrics = {
                'silhouette': float(silhouette_score(X_test, y_pred)),
                'calinski_harabasz': float(calinski_harabasz_score(X_test, y_pred)),
                'davies_bouldin': float(davies_bouldin_score(X_test, y_pred)),
                'inertia': float(model.inertia_),
            }
            
            # Multiclass evaluation using majority vote with the trained model
            y_pred_class, mapping = kmeans_eval(model, X_test, y_test)
            
            # Multiclass metrics
            metrics = {
                **clustering_metrics,
                'accuracy': float(accuracy_score(y_test, y_pred_class)),
                'precision_weighted': float(precision_score(y_test, y_pred_class, average='weighted', zero_division=0)),
                'recall_weighted': float(recall_score(y_test, y_pred_class, average='weighted', zero_division=0)),
                'f1_weighted': float(f1_score(y_test, y_pred_class, average='weighted', zero_division=0)),
                'confusion_matrix': confusion_matrix(y_test, y_pred_class),
                'cluster_label_map': mapping,
            }
            
            # Add per-class metrics for K-means too
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                y_test, y_pred_class, average=None, zero_division=0
            )
            
            metrics['precision_per_class'] = precision_per_class.tolist()
            metrics['recall_per_class'] = recall_per_class.tolist()
            metrics['f1_per_class'] = f1_per_class.tolist()
            metrics['support_per_class'] = support_per_class.tolist()
            
            # Try to calculate multiclass AUC (one-vs-rest)
            try:
                # For multiclass AUC, we need probability scores
                # Since K-means doesn't provide probabilities, we'll skip this
                metrics['roc_auc_ovr'] = float('nan')
            except:
                metrics['roc_auc_ovr'] = float('nan')
                
        else:  # Supervised: multiclass classification
            y_pred = model.predict(X_test)
            
            # Get probabilities if available
            try:
                y_proba = model.predict_proba(X_test)
                # Multiclass AUC (one-vs-rest)
                roc_auc_ovr = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
            except:
                y_proba = None
                roc_auc_ovr = float('nan')
            
            # Multiclass metrics
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                'roc_auc_ovr': float(roc_auc_ovr),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
            }
            
            # Add per-class metrics
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                y_test, y_pred, average=None, zero_division=0
            )
            
            metrics['precision_per_class'] = precision_per_class.tolist()
            metrics['recall_per_class'] = recall_per_class.tolist()
            metrics['f1_per_class'] = f1_per_class.tolist()
            metrics['support_per_class'] = support_per_class.tolist()

        results[name] = metrics

    return results

def calculate_label_metrics(models: Dict[str, object], X_test, y_test, traffic_types: list) -> Dict[str, Dict[str, float]]:
    """
    Calculate Label metrics from Traffic Type predictions for all models
    """
    traffic_type_to_label_map = {
        'Audio': 0, 'Background': 0, 'Text': 0, 'Video': 0,  # Benign types
        'Bruteforce': 1, 'DoS': 1, 'Information_Gathering': 1, 'Mirai': 1  # Malicious types
    }
    
    def to_label(name: str) -> int:
        return traffic_type_to_label_map.get(name, 0)  # Default to benign if unknown
    
    # Calculate Label metrics for all models
    all_label_metrics = {}
    classes = traffic_types
    
    # Get true labels once
    y_true_names = [classes[i] for i in y_test]
    y_true_label = [to_label(n) for n in y_true_names]
    
    for model_name, model in models.items():
        # Predict Traffic Type for this model
        y_pred_type = model.predict(X_test)
        
        # Map indices -> names
        y_pred_names = [classes[i] for i in y_pred_type]
        
        # Convert to binary labels
        y_pred_label = [to_label(n) for n in y_pred_names]
        
        # Calculate metrics
        label_metrics = {
            'accuracy': float(accuracy_score(y_true_label, y_pred_label)),
            'precision': float(precision_score(y_true_label, y_pred_label, zero_division=0)),
            'recall': float(recall_score(y_true_label, y_pred_label, zero_division=0)),
            'f1': float(f1_score(y_true_label, y_pred_label, zero_division=0))
        }
        
        all_label_metrics[model_name] = label_metrics
    
    return all_label_metrics

def print_results(results: Dict[str, Dict[str, object]], traffic_types: list):
    """
    Print formatted multiclass evaluation results
    """
    print("\n" + "="*80)
    print("MULTICLASS THREAT CLASSIFICATION RESULTS")
    print("="*80)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print("-" * 40)
        
        if 'accuracy' in metrics:
            print(f"Accuracy: {metrics['accuracy']:.4f}")
        
        if 'precision_weighted' in metrics:
            print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
        
        if 'recall_weighted' in metrics:
            print(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")
        
        if 'f1_weighted' in metrics:
            print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        
        if 'roc_auc_ovr' in metrics and not np.isnan(metrics['roc_auc_ovr']):
            print(f"ROC AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
        
        # Clustering metrics for K-means
        if 'silhouette' in metrics:
            print(f"Silhouette Score: {metrics['silhouette']:.4f}")
        if 'calinski_harabasz' in metrics:
            print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz']:.4f}")
        if 'davies_bouldin' in metrics:
            print(f"Davies-Bouldin Score: {metrics['davies_bouldin']:.4f}")
        if 'inertia' in metrics:
            print(f"Inertia: {metrics['inertia']:.4f}")
        
        # Per-class metrics if available
        if 'precision_per_class' in metrics:
            print(f"\nPer-Class Metrics:")
            for i, traffic_type in enumerate(traffic_types):
                if i < len(metrics['precision_per_class']):
                    print(f"  {traffic_type}:")
                    print(f"    Precision: {metrics['precision_per_class'][i]:.4f}")
                    print(f"    Recall: {metrics['recall_per_class'][i]:.4f}")
                    print(f"    F1-Score: {metrics['f1_per_class'][i]:.4f}")
                    print(f"    Support: {metrics['support_per_class'][i]}")

def print_label_results(label_metrics: Dict[str, Dict[str, float]]):
    """
    Print Label metrics results
    """
    print("\nLabel Metrics (from Traffic Type):")
    print("="*60)
    
    for model_name, metrics in label_metrics.items():
        print(f"\n{model_name.upper()}:")
        print(f"\tAccuracy: {metrics['accuracy']:.4f}")
        print(f"\tPrecision: {metrics['precision']:.4f}")
        print(f"\tRecall: {metrics['recall']:.4f}")
        print(f"\tF1-Score: {metrics['f1']:.4f}")
