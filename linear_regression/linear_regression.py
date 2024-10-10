import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processing import train_test_split, StandardScaler
from evaluation_methods import mean_absolute_error, mean_squared_error

class LinearRegression:
	def __init__(self):
		self.theta_0_ = 0
		self.theta_1_ = 0
		self.learning_rate_ = 0.1
		self.iterations_ = 1000
		self.random_seed = 42
		self.X_ = None
		self.y_ = None
		self.X_train_scaled_ = None
		self.X_test_ = None
		self.y_train_ = None
		self.y_test_ = None
		self.scaler_ = None

	def predict(self, prediction_input):
		return self.theta_0_ + self.theta_1_ * prediction_input
	
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
		for _ in range(self.iterations_):
			self.theta_0_, self.theta_1_ = self.update_thetas()
		self.transform_thetas_to_original_units()

	def load_data(self):
		df = pd.read_csv('data/data.csv')
		self.X_ = df.drop('price', axis=1)
		self.y_ = df['price']

	def split_data(self):
		X_train, X_test, self.y_train_, self.y_test_ = train_test_split(self.X_,
																																		self.y_,
																																		test_size=0.2,
																																		random_state=self.random_seed)
		self.X_train = np.array(X_train, dtype=np.float64).squeeze()
		self.X_test_ = np.array(X_test, dtype=np.float64).squeeze()
		self.y_train_ = np.array(self.y_train_, dtype=np.float64).squeeze()
		self.y_test_ = np.array(self.y_test_, dtype=np.float64).squeeze()

	def standarize_data(self):
		self.scaler_ = StandardScaler()
		self.X_train_scaled_ = self.scaler_.fit_transform(self.X_train)

	def update_thetas(self):
		constant = self.learning_rate_ / len(self.X_train_scaled_)
		error = self.predict(self.X_train_scaled_) - self.y_train_
		tmp_theta_0 = constant * np.sum(error)
		tmp_theta_1 = constant * np.sum(error * self.X_train_scaled_)
		return self.theta_0_ - tmp_theta_0, self.theta_1_ - tmp_theta_1
	
	def transform_thetas_to_original_units(self):
		mu = self.scaler_.mean_
		sigma = self.scaler_.std_
		self.theta_0_ = self.theta_0_ - np.sum((mu / sigma) * self.theta_1_)
		self.theta_1_ = self.theta_1_ / sigma

	def evaluate_model(self):
		y_pred = self.predict(self.X_test_)
		rmse = np.sqrt(mean_squared_error(y_pred, self.y_test_))
		y_test_mean = np.mean(self.y_test_)
		mae = mean_absolute_error(y_pred, self.y_test_)
		error_percentage = 100 * mae / y_test_mean
		return rmse, y_test_mean, mae, error_percentage

	def plot_results(self):
		X = np.array(self.X_, dtype=float)
		y = np.array(self.y_, dtype=float)

		fig, ax = plt.subplots()
		ax.scatter(X, y, color='blue')
		X_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
		y_range_pred = self.theta_1_ * X_range + self.theta_0_
		ax.plot(X_range, y_range_pred, color='red', label='Tendency line')
		ax.set_xlabel('mileage')
		ax.set_ylabel('price')
		ax.set_title('Linear Regression')
		ax.legend()
		return fig, ax
