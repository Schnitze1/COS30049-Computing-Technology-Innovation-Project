from typing import Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

def evaluate_models(models: Dict[str, object], X_test, y_test) -> Dict[str, Dict[str, object]]:
	results: Dict[str, Dict[str, object]] = {}
	for name, model in models.items():
		y_pred = model.predict(X_test)
		proba = model.predict_proba(X_test)

		# Build metrics first
		metrics = {
			'accuracy': float(accuracy_score(y_test, y_pred)),
			'f1': float(f1_score(y_test, y_pred, average='binary')),
			'precision': float(precision_score(y_test, y_pred, average='binary', zero_division=0)),
			'recall': float(recall_score(y_test, y_pred, average='binary')),
   			'roc_auc': float(roc_auc_score(y_test, proba[:, 1])),
			'confusion_matrix': confusion_matrix(y_test, y_pred),
		}

		results[name] = metrics

	return results

