def display_banner():
	print(
		"\n"
		"////////////////////////////////////////////////////////////////////\n\n"
		"                        ft_linear_regression                        \n\n"
		"////////////////////////////////////////////////////////////////////")

def display_welcome_message():
	print(
		"\nThis program uses a linear regression model to estimate the worth of a"
		"car given its mileage.\n"
		"The model is trained using pre-loaded data.\n"
		"The initial values for the intercept and slope parameter are zero.\n"
		"The generated graphs can be found within the ./graphs/ directory.")

def ask_for_commands():
	print("\nCommands available: 'estimate' 'plot_data' 'train' 'exit'.")

def display_not_a_number_warning():
	print("\nPlease provide a number.")

def display_negative_estimation_input_warning():
	print("\nThe mileage given was negative.")
	
def display_estimation(estimation):
	print(f"\nThe estimated value of the car is: {round(estimation, 2)}")
	if estimation < 0:
		print("\nWarning: The estimated value is negative.")

def display_training_start_status():
	print("\nTraining model...")

def display_training_end_status():
	print("\nTraining complete.\n")

def display_training_results(rmse, y_mean, mae, error_percentage):
	print(f"Root Mean Squared Error: {round(rmse, 2)}")
	print(f"Mean value of price: {round(y_mean, 2)}")
	print(f"Mean Absolute Error: {round(mae, 2)}")
	print(f"Error Percentage: {round(error_percentage, 2)}%")