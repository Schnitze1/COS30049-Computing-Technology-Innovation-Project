from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

def train_models(X_train, y_train, n_classes: int) -> Dict[str, object]:
    """
    Train models for multiclass classification
    """
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        ),
        'mlp': MLPClassifier(
            hidden_layer_sizes=(100, 100),
            solver='adam',
            learning_rate_init=1e-3,
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=10,
            tol=1e-4,
            random_state=42
        ),
        'kmeans': KMeans(n_clusters=n_classes, random_state=42),
    }

    for name, model in models.items():
        print(f"Training {name}...")
        if name == 'kmeans':  # Unsupervised: clustering
            model.fit(X_train)
        else:  # Supervised: classification
            model.fit(X_train, y_train)

    return models
