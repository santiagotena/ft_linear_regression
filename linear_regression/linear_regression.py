import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processing import train_test_split, StandardScaler
from evaluation_methods import mean_absolute_error, mean_squared_error

class LinearRegression:
	def __init__(self):
		self.theta_0 = 0
		self.theta_1 = 0
		self.learning_rate = 0.1
		self.iterations = 1000
		self.X = None
		self.y = None
		self.X_train_scaled = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
		self.scaler = None

	def predict(self, prediction_input):
		return self.theta_0 + self.theta_1 * prediction_input
	
	def plot_data(self):
		df = pd.read_csv('data/data.csv')
		X = df.drop('price', axis=1)
		y = df['price']

		fig, ax = plt.subplots()
		ax.scatter(X, y, color='blue')
		ax.set_xlabel('mileage')
		ax.set_ylabel('price')
		ax.set_title('Data')
		return fig, ax

	def train(self):
		self.load_data()
		self.split_data()
		self.standarize_data()
		for _ in range(self.iterations):
			self.theta_0, self.theta_1 = self.update_thetas()
		self.transform_thetas_to_original_units()

	def load_data(self):
		df = pd.read_csv('data/data.csv')
		self.X = df.drop('price', axis=1)
		self.y = df['price']
	
	def split_data(self):
		X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
		self.X_train = np.array(X_train, dtype=np.float64).squeeze()
		self.X_test = np.array(X_test, dtype=np.float64).squeeze()
		self.y_train = np.array(self.y_train, dtype=np.float64).squeeze()
		self.y_test = np.array(self.y_test, dtype=np.float64).squeeze()

	def standarize_data(self):
		self.scaler = StandardScaler()
		self.X_train_scaled = self.scaler.fit_transform(self.X_train)

	def update_thetas(self):
		constant = self.learning_rate / len(self.X_train_scaled)
		error = self.predict(self.X_train_scaled) - self.y_train
		tmp_theta_0 = constant * np.sum(error)
		tmp_theta_1 = constant * np.sum(error * self.X_train_scaled)
		return self.theta_0 - tmp_theta_0, self.theta_1 - tmp_theta_1
	
	def transform_thetas_to_original_units(self):
		mu = self.scaler.mean_
		sigma = self.scaler.std_
		self.theta_0 = self.theta_0 - np.sum((mu / sigma) * self.theta_1)
		self.theta_1 = self.theta_1 / sigma
	
	def evaluate_model(self):
		y_pred = self.predict(self.X_test)
		rmse = np.sqrt(mean_squared_error(y_pred, self.y_test))
		y_mean = np.mean(self.y)
		mae = mean_absolute_error(y_pred, self.y_test)
		error_percentage = 100 * mae / y_mean
		return rmse, y_mean, mae, error_percentage

	def plot_results(self):
		X = np.array(self.X, dtype=float)
		y = np.array(self.y, dtype=float)

		fig, ax = plt.subplots()
		ax.scatter(X, y, color='blue')
		X_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
		y_range_pred = self.theta_1 * X_range + self.theta_0
		ax.plot(X_range, y_range_pred, color='red', label='Tendency line')
		ax.set_xlabel('mileage')
		ax.set_ylabel('price')
		ax.set_title('Linear Regression')
		ax.legend()
		return fig, ax
