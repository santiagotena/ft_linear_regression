from .interface_messages import *
from .utils import *

def handle_train_command(linear_regression):
	display_training_start_status()
	linear_regression.train()
	display_training_end_status()
	present_training_results(linear_regression)
	plot_results(linear_regression)

def present_training_results(linear_regression):
	rmse, y_mean, mae, error_percentage = linear_regression.evaluate_model()
	display_training_results(rmse, y_mean, mae, error_percentage)

def plot_results(linear_regression):
	linear_regression.plot_results()
	store_plot("linear_regression.png")