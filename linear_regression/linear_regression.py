import os

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

		plt.scatter(X, y, color='blue')
		plt.xlabel('mileage')
		plt.ylabel('price')
		plt.title('Data')
		if not os.path.exists("graphs"):
			os.makedirs("graphs")
		plt.savefig("graphs/data_scatterplot.png")
		plt.show()

	def train(self):
		df = pd.read_csv('data/data.csv')
		self.X = df.drop('price', axis=1)
		self.y = df['price']

		X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
		X_train = np.array(X_train, dtype=np.float64).squeeze()
		self.X_test = np.array(X_test, dtype=np.float64).squeeze()
		self.y_train = np.array(self.y_train, dtype=np.float64).squeeze()
		self.y_test = np.array(self.y_test, dtype=np.float64).squeeze()

		self.scaler = StandardScaler()
		self.X_train_scaled = self.scaler.fit_transform(X_train)
		# self.X_test_scaled = self.scaler.fit_transform(X_test)

		for _ in range(self.iterations):
			self.theta_0, self.theta_1 = self.update_thetas()

		mu = self.scaler.mean
		sigma = self.scaler.std
		self.theta_0 = self.theta_0 - np.sum((mu / sigma) * self.theta_1)
		self.theta_1 = self.theta_1 / sigma

	def update_thetas(self):
		tmp_theta_0 = self.learning_rate/len(self.X_train_scaled) * np.sum((self.predict(self.X_train_scaled) - self.y_train))
		tmp_theta_1 = self.learning_rate/len(self.X_train_scaled) * np.sum((self.predict(self.X_train_scaled) - self.y_train) * self.X_train_scaled)
		return self.theta_0 - tmp_theta_0, self.theta_1 - tmp_theta_1
	
	def evaluate_model(self):
		y_pred = self.predict(self.X_test)
		rmse = np.sqrt(mean_squared_error(y_pred, self.y_test))
		y_mean = np.mean(self.y)
		mae = mean_absolute_error(y_pred, self.y_test)
		error_percentage = 100 * mae / y_mean
		return rmse, y_mean, mae, error_percentage
		print("Root Mean Squared Error:", rmse)
		print("Mean value of price:", y_mean)
		print("Mean Absolute Error:", mae)
		print("Error Percentage:", error_percentage)

	def plot_results(self):
		X = np.array(self.X, dtype=float)
		y = np.array(self.y, dtype=float)

		plt.scatter(X, y, color='blue')
		X_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
		y_range_pred = self.theta_1 * X_range + self.theta_0
		plt.plot(X_range, y_range_pred, color='red', label='Tendency line')
		plt.xlabel('mileage')
		plt.ylabel('price')
		plt.title('Linear Regression')
		plt.legend()
		if not os.path.exists("graphs"):
			os.makedirs("graphs")
		plt.savefig("graphs/linear_regression.png")
		plt.show()
