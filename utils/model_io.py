import os
from typing import Dict
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib  # type: ignore

def save_models(models: Dict[str, object], out_dir: str = "cache/models") -> None:
    for name, model in models.items():
        if isinstance(model, (RandomForestClassifier, MLPClassifier)) or isinstance(model, BaseEstimator):
            model_path = os.path.join(out_dir, f"{name}.joblib")
            joblib.dump(model, model_path)
        else:
            raise ValueError(f"Unsupported model type for saving: {type(model)}")


def load_model(name: str, out_dir: str = "cache/models") -> object:
    model_path = os.path.join(out_dir, f"{name}.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    raise FileNotFoundError(f"Model '{name}' not found in {out_dir}")


def list_models(out_dir: str = "cache/models") -> Dict[str, str]:
    if not os.path.exists(out_dir):
        return {}
    models: Dict[str, str] = {}
    for fname in os.listdir(out_dir):
        if fname.endswith(".joblib"):
            name = fname[:-7]
            models[name] = "sklearn"
    return models


