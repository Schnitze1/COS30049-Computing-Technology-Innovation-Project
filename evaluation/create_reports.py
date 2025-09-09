import os
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

import os
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def results_to_dataframe(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
	rows = []
	for model_name, metrics in results.items():
		row = { 'model': model_name }
		row.update(metrics)
		rows.append(row)
	return pd.DataFrame(rows)


def save_results_csv(results: Dict[str, Dict[str, float]], out_dir: str = 'evaluation_reports/multiclass', filename: str = 'metrics.csv') -> str:
	df = results_to_dataframe(results)
	out_path = os.path.join(out_dir, filename)
	df.to_csv(out_path, index=False)
	return out_path

def plot_confusion_matrices(results: Dict[str, Dict[str, float]], out_dir: str = 'evaluation_reports/multiclass', 
                          class_labels: Optional[List[str]] = None) -> str:
	items = [(name, res['confusion_matrix']) for name, res in results.items() if isinstance(res, dict) and 'confusion_matrix' in res]
	if not items:
		raise ValueError('No confusion matrices available')
	cols = min(3, len(items))
	rows = (len(items) + cols - 1) // cols
	plt.figure(figsize=(5*cols, 4*rows))

	for idx, (name, cm) in enumerate(items, start=1):
		ax = plt.subplot(rows, cols, idx)
		
		# Determine if binary or multiclass
		if hasattr(cm, 'shape') and cm.shape == (2, 2):
			# Binary classification
			labels = ['False', 'True'] if class_labels is None else class_labels[:2]
			sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
					xticklabels=labels, yticklabels=labels, ax=ax)
		else:
			# Multiclass classification
			if class_labels is not None and len(class_labels) == cm.shape[0]:
				labels = class_labels
			else:
				labels = [f'Class {i}' for i in range(cm.shape[0])]
			
			# For large confusion matrices, use smaller annotations
			annot = True if cm.shape[0] <= 10 else False
			sns.heatmap(cm, annot=annot, fmt='d', cmap='Blues', cbar=True, 
					xticklabels=labels, yticklabels=labels, ax=ax)
		
		ax.set_title(name)
		ax.set_xlabel('Predicted')
		ax.set_ylabel('True')

	plt.tight_layout()
	out_path = os.path.join(out_dir, 'confusion_matrices.png')
	plt.savefig(out_path, dpi=150)
	plt.close()
	return out_path

def plot_multiclass_metrics(results: Dict[str, Dict[str, float]], out_dir: str = 'evaluation_reports/multiclass') -> str:
	"""Create comprehensive multiclass metrics visualization"""
	df = results_to_dataframe(results)
	
	# Define multiclass metrics to plot (using weighted averages for better imbalanced dataset representation)
	multiclass_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovr']
	available_metrics = [m for m in multiclass_metrics if m in df.columns]
	
	if not available_metrics:
		raise ValueError('No multiclass metrics available')
	
	rows, cols = len(available_metrics), 1
	plt.figure(figsize=(8, 3*len(available_metrics)))

	for idx, metric in enumerate(available_metrics, start=1):
		ax = plt.subplot(rows, cols, idx)
		plot_df = df[['model', metric]].copy()
		plot_df = plot_df.sort_values(metric, ascending=False)
		
		bars = ax.bar(plot_df['model'], plot_df[metric])
		ax.set_title(f'{metric.replace("_", " ").title()}')
		ax.set_ylabel(metric.replace("_", " ").title())
		ax.set_xlabel('Model')
		
		# Color bars based on performance
		for i, bar in enumerate(bars):
			value = plot_df[metric].iloc[i]
			if value >= 0.9:
				bar.set_color('green')
			elif value >= 0.8:
				bar.set_color('orange')
			else:
				bar.set_color('red')
		
		# Add value annotations
		for i, v in enumerate(plot_df[metric].tolist()):
			ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom')
		
		# Set ticks and labels properly
		ax.set_xticks(range(len(plot_df['model'])))
		ax.set_xticklabels(plot_df['model'], rotation=45, ha='right')
		ax.set_ylim(0, 1.1)

	plt.tight_layout()
	out_path = os.path.join(out_dir, 'multiclass_metrics_comparison.png')
	plt.savefig(out_path, dpi=150)
	plt.close()
	return out_path

def plot_per_class_metrics(results: Dict[str, Dict[str, float]], traffic_types: List[str], 
                          out_dir: str = 'evaluation_reports/multiclass') -> List[str]:
	"""Plot per-class metrics for all models with per-class metrics"""
	paths = []
	
	# Find all models with per-class metrics
	models_with_per_class = []
	for model_name, metrics in results.items():
		if 'precision_per_class' in metrics:
			models_with_per_class.append(model_name)
	
	if not models_with_per_class:
		raise ValueError('No per-class metrics available')
	
	# Create plots for each model
	for model_name in models_with_per_class:
		metrics_data = results[model_name]
		precision_per_class = metrics_data['precision_per_class']
		recall_per_class = metrics_data['recall_per_class']
		f1_per_class = metrics_data['f1_per_class']
		
		# Create DataFrame for plotting
		df_per_class = pd.DataFrame({
			'Traffic_Type': traffic_types,
			'Precision': precision_per_class,
			'Recall': recall_per_class,
			'F1_Score': f1_per_class
		})
		
		# Melt for easier plotting
		df_melted = df_per_class.melt(id_vars=['Traffic_Type'], 
		                             value_vars=['Precision', 'Recall', 'F1_Score'],
		                             var_name='Metric', value_name='Score')
		
		plt.figure(figsize=(12, 6))
		sns.barplot(data=df_melted, x='Traffic_Type', y='Score', hue='Metric')
		plt.title(f'Per-Class Metrics - {model_name.upper()}')
		plt.xlabel('Traffic Type')
		plt.ylabel('Score')
		plt.xticks(rotation=45, ha='right')
		plt.legend(title='Metric')
		plt.tight_layout()
		
		out_path = os.path.join(out_dir, f'per_class_metrics_{model_name}.png')
		plt.savefig(out_path, dpi=150)
		plt.close()
		paths.append(out_path)
	
	return paths

def save_label_metrics(label_metrics: Dict[str, Dict[str, float]], out_dir: str = 'evaluation_reports/binary_label') -> str:
	"""Save label metrics to CSV"""
	# Convert to DataFrame format
	rows = []
	for model_name, metrics in label_metrics.items():
		row = {'model': model_name}
		row.update(metrics)
		rows.append(row)
	
	df = pd.DataFrame(rows)
	out_path = os.path.join(out_dir, 'label_from_type_metrics.csv')
	df.to_csv(out_path, index=False)
	return out_path

def export_multiclass_all(results: Dict[str, Dict[str, float]], traffic_types: List[str], 
                         label_metrics: Dict[str, Dict[str, float]] = None,
                         out_dir: str = 'evaluation_reports') -> Dict[str, str]:
	"""Export all multiclass reports and visualizations"""
	paths = {}
	multiclass_out_dir = os.path.join(out_dir, 'multiclass')
	binary_label_out_dir = os.path.join(out_dir, 'binary_label')
	
	# Save summary CSV
	paths['Multiclass Summary CSV'] = save_results_csv(results, multiclass_out_dir, 'multiclass_metrics_summary.csv')
	
	# Save label metrics if provided
	if label_metrics:
		paths['Label-from-type CSV'] = save_label_metrics(label_metrics, binary_label_out_dir)
	
	# Create visualizations
	try:
		paths['Multiclass Metrics Comparison'] = plot_multiclass_metrics(results, multiclass_out_dir)
	except Exception as e:
		print(f"Warning: Could not create multiclass metrics plot: {e}")
	
	try:
		paths['Confusion Matrices'] = plot_confusion_matrices(results, multiclass_out_dir, traffic_types)
	except Exception as e:
		print(f"Warning: Could not create confusion matrices: {e}")
	
	try:
		per_class_paths = plot_per_class_metrics(results, traffic_types, multiclass_out_dir)
		for i, path in enumerate(per_class_paths):
			model_name = path.split('_')[-1].replace('.png', '')
			paths[f'Per-Class Metrics ({model_name})'] = path
	except Exception as e:
		print(f"Warning: Could not create per-class metrics plots: {e}")
	
	return paths

def plot_kmeans_pca_scatter(X: np.ndarray, y_true: np.ndarray, y_clusters: np.ndarray,
                          traffic_types: List[str], out_dir: str = 'evaluation_reports/clustering',) -> Dict[str, str]:
	"""Create a PCA scatter plot colored by K-Means cluster.

	Parameters
	----------
	X : np.ndarray
		Feature matrix used for evaluation (test set).
	y_true : np.ndarray
		True class indices for the samples (unused here, kept for API symmetry).
	y_clusters : np.ndarray
		K-Means cluster assignments for the samples.
	traffic_types : List[str]
		Class label names (unused here, kept for API symmetry).
	out_dir : str
		Directory to save plots.
	filename_prefix : str
		Prefix for output filenames.

	Returns
	-------
	Dict[str, str]
		Mapping of plot description to saved file paths.
	"""
	# Reduce to 2D with PCA for visualization
	pca = PCA(n_components=2, random_state=42)
	X_2d = pca.fit_transform(X)

	# Plot colored by K-Means cluster
	plt.figure(figsize=(8, 6))
	scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_clusters, cmap='tab20', s=10, alpha=0.7)
	plt.title('K-Means Clusters (PCA 2D)')
	plt.xlabel('PC1')
	plt.ylabel('PC2')
	cbar = plt.colorbar(scatter, boundaries=np.arange(int(np.max(y_clusters)) + 2) - 0.5)
	cbar.set_label('Cluster ID')
	cluster_path = os.path.join(out_dir, f'kmeans_pca_by_cluster.png')
	plt.tight_layout()
	plt.savefig(cluster_path, dpi=150)
	plt.close()

	return {
		'KMeans PCA by Cluster': cluster_path,
	}


def plot_cluster_label_heatmap(y_clusters: np.ndarray, y_true: np.ndarray,
                             traffic_types: List[str], out_dir: str = 'evaluation_reports/clustering',) -> str:
	"""Heatmap of cluster vs true label counts (contingency matrix)."""
	# Ensure integers
	y_clusters = np.asarray(y_clusters).astype(int)
	y_true = np.asarray(y_true).astype(int)

	n_clusters = int(np.max(y_clusters)) + 1
	matrix = np.zeros((n_clusters, len(traffic_types)), dtype=int)
	for c in range(n_clusters):
		index = np.where(y_clusters == c)[0]
		if index.size == 0:
			continue
		vals, counts = np.unique(y_true[index], return_counts=True)
		for val, count in zip(vals, counts):
			matrix[c, int(val)] = int(count)

	plt.figure(figsize=(max(8, len(traffic_types)*0.9), max(5, n_clusters*0.4)))
	sns.heatmap(matrix, annot=False, fmt='d', cmap='Purples', cbar=True,
	            yticklabels=[f'C{c}' for c in range(n_clusters)],
	            xticklabels=traffic_types)
	plt.xlabel('True Label')
	plt.ylabel('Cluster ID')
	plt.title('Cluster vs True Label Counts')
	out_path = os.path.join(out_dir, 'kmeans_cluster_label_heatmap.png')
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()
	return out_path


