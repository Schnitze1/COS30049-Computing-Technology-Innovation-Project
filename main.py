import argparse
import os
from train import train_baseline_models
from evaluation.calc_eval_metrics import evaluate_models
from evaluation.create_reports import export_all
import numpy as np
import pickle
from utils.model_io import save_models

DATA_PATH = 'data_prerprocessing/output'

def load_dataset(npz_path: str = f'{DATA_PATH}/processed_data.npz'):
	data = np.load(npz_path)
	X_train = data['X_train']
	X_test = data['X_test']
	y_train = data['y_train']
	y_test = data['y_test']
	return X_train, X_test, y_train, y_test


def load_feature_metadata(pickle_path: str = f'{DATA_PATH}/feature_metadata.pkl'):
	try:
		with open(pickle_path, 'rb') as f:
			return pickle.load(f)
	except FileNotFoundError:
		return None


def parse_args():
    parser = argparse.ArgumentParser(description='Simple classification evaluation runner')
    parser.add_argument('--data', type=str, default=f'{DATA_PATH}/processed_data.npz')
    parser.add_argument('--features', type=str, default=f'{DATA_PATH}/feature_metadata.pkl')
    return parser.parse_args()


def main():
    for dir in ['cache/models', 'evaluation_reports']:
        os.makedirs(dir, exist_ok=True)
        
    args = parse_args()
    X_train, X_test, y_train, y_test = load_dataset(args.data)
    
    # We'll use them later for plotting (convert encoded metadata back to their original string names)
    # _ = load_feature_metadata(args.features)
    
    models = train_baseline_models(X_train, y_train)
    # Save trained models for later inference
    save_models(models, out_dir='cache/models')
    
    results = evaluate_models(models, X_test, y_test)
    for name, metrics in results.items():
        print(name)
        for metric, value in metrics.items():
            print(f'\t{metric}: {value}')
    paths = export_all(results, out_dir='evaluation_reports')
    
    print('Artifacts saved to:')
    for path in paths:
        print(f'\t{path}')

if __name__ == "__main__":
    main()