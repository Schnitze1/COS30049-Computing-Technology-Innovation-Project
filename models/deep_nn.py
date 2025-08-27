import numpy as np
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore
from sklearn.base import BaseEstimator, ClassifierMixin


class SimpleNeuralNetwork(BaseEstimator, ClassifierMixin):
	"""Simple neural network"""
	
	def __init__(self, hidden_layers=(64, 32), dropout_rate=0.2, epochs=50, batch_size=32, random_state=42):
		self.hidden_layers = hidden_layers
		self.dropout_rate = dropout_rate
		self.epochs = epochs
		self.batch_size = batch_size
		self.random_state = random_state
		self.model = None
		self.classes_ = None
	
	def fit(self, X, y):
		# Set random seeds for reproducibility
		np.random.seed(self.random_state)
		keras.utils.set_random_seed(self.random_state)
		
		# Store classes
		self.classes_ = np.unique(y)
		
		# Build model
		self.model = keras.Sequential()
		
		# Input layer
		self.model.add(keras.Input(shape=(X.shape[1],)))
		self.model.add(layers.Dense(self.hidden_layers[0], activation='relu'))
		self.model.add(layers.Dropout(self.dropout_rate))
		
		# Hidden layers
		for units in self.hidden_layers[1:]:
			self.model.add(layers.Dense(units, activation='relu'))
			self.model.add(layers.Dropout(self.dropout_rate))
		
		# Output layer (binary classification)
		self.model.add(layers.Dense(1, activation='sigmoid'))
		
		# Compile
		self.model.compile(
			optimizer='adam',
			loss='binary_crossentropy',
			metrics=['accuracy']
		)
		
		# Train
		self.model.fit(
			X, y,
			epochs=self.epochs,
			batch_size=self.batch_size,
			verbose=0
		)
		
		return self
	
	def predict(self, X):
		predictions = self.model.predict(X, verbose=0)
		return (predictions > 0.5).astype(int).flatten()
	
	def predict_proba(self, X):
		proba = self.model.predict(X, verbose=0)
		# Return shape (n_samples, 2) for binary classification
		return np.column_stack([1 - proba, proba])
	
	def get_params(self):
		return {
			'hidden_layers': self.hidden_layers,
			'dropout_rate': self.dropout_rate,
			'epochs': self.epochs,
			'batch_size': self.batch_size,
			'random_state': self.random_state
		}
	
	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self
