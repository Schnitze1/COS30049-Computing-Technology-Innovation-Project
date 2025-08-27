import os
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def results_to_dataframe(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
	rows = []
	for model_name, metrics in results.items():
		row = { 'model': model_name }
		row.update(metrics)
		rows.append(row)
	return pd.DataFrame(rows)


def save_results_csv(results: Dict[str, Dict[str, float]], out_dir: str = 'evaluation_reports', filename: str = 'metrics.csv') -> str:
	df = results_to_dataframe(results)
	out_path = os.path.join(out_dir, filename)
	df.to_csv(out_path, index=False)
	return out_path


def plot_metric_bar(results: Dict[str, Dict[str, float]], metric: str, out_dir: str = 'evaluation_reports', sort_desc: bool = False) -> str:
	df = results_to_dataframe(results)
	if metric not in df.columns:
		raise ValueError(f"Metric '{metric}' not found in results")
	plot_df = df[['model', metric]].copy()
	if sort_desc:
		plot_df = plot_df.sort_values(metric, ascending=False)
	plt.figure(figsize=(10, 5))
	plt.bar(plot_df['model'], plot_df[metric])
	plt.xlabel('Model')
	plt.ylabel(metric.upper())
	plt.title(f"{metric.upper()} Comparison" + (" (Sorted)" if sort_desc else ""))
	plt.xticks(rotation=45, ha='right')
	# annotate values
	for i, v in enumerate(plot_df[metric].tolist()):
		plt.text(i, v + 0.005, f"{v:.4f}", ha='center', va='bottom')
	plt.tight_layout()
	name = f"{metric}_bar{'_sorted' if sort_desc else ''}.png"
	out_path = os.path.join(out_dir, name)
	plt.savefig(out_path, dpi=150)
	plt.close()
	return out_path


def export_all(results: Dict[str, Dict[str, float]], out_dir: str = 'evaluation_reports') -> Dict[str, str]:
	paths = {}
	paths['Metrics CSV'] = save_results_csv(results, out_dir)
	paths['Bar charts'] = plot_metrics_bar_charts(results, out_dir=out_dir, sort_desc=True)
	paths['Confusion matrices'] = plot_confusion_matrices(results, out_dir)
	return paths


def plot_metrics_bar_charts(results: Dict[str, Dict[str, float]], out_dir: str = 'evaluation_reports', sort_desc: bool = True) -> str:
	"""
		Create a single figure with bar charts for f1, roc_auc, accuracy, precision, recall.
	"""
	df = results_to_dataframe(results)
	metrics = ['f1', 'roc_auc', 'accuracy', 'precision', 'recall']

	rows, cols = len(metrics), 1
	plt.figure(figsize=(6, 3.5*len(metrics)))

	for idx, metric in enumerate(metrics, start=1):
		if metric not in df.columns:
			continue
		ax = plt.subplot(rows, cols, idx)
		plot_df = df[['model', metric]].copy()
		if sort_desc:
			plot_df = plot_df.sort_values(metric, ascending=False)
		ax.bar(plot_df['model'], plot_df[metric])
		ax.set_title(metric.upper())
		ax.set_ylabel(metric.upper())
		ax.set_xlabel('Model')

		# Set ticks before labels 
		tick_positions = list(range(len(plot_df['model'])))
		ax.set_xticks(tick_positions)
		ax.set_xticklabels(plot_df['model'], rotation=45, ha='right')
		for i, v in enumerate(plot_df[metric].tolist()):
			ax.text(i, v + 0.005, f"{v:.3f}", ha='center', va='bottom')

	plt.tight_layout()
	name = f"all_metrics_bar_charts.png"
	out_path = os.path.join(out_dir, name)
	plt.savefig(out_path, dpi=150)
	plt.close()
	return out_path


def plot_confusion_matrices(results: Dict[str, Dict[str, float]], out_dir: str = 'evaluation_reports') -> str:
	items = [(name, res['confusion_matrix']) for name, res in results.items() if isinstance(res, dict) and 'confusion_matrix' in res]
	if not items:
		raise ValueError('No confusion matrices available')
	cols = min(3, len(items))
	rows = (len(items) + cols - 1) // cols
	plt.figure(figsize=(5*cols, 4*rows))

	for idx, (name, cm) in enumerate(items, start=1):
		ax = plt.subplot(rows, cols, idx)
		if hasattr(cm, 'shape') and cm.shape == (2, 2):
			sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
					xticklabels=['False', 'True'], yticklabels=['False', 'True'], ax=ax)
			ax.set_title(name)
		else:
			sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True, ax=ax)
			ax.set_title(name)
		ax.set_xlabel('Predicted')
		ax.set_ylabel('True')

	plt.tight_layout()
	out_path = os.path.join(out_dir, 'confusion_matrices.png')
	plt.savefig(out_path, dpi=150)
	plt.close()
	return out_path
