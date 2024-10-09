from linear_regression import LinearRegression

def ask_user_input():
	show_banner()
	show_welcome_message()
	linear_regression = LinearRegression()
	while(True):
		ask_for_commands()
		command_input = input("Enter a command: ")
		if command_input == "estimate":
			process_estimate_command(linear_regression)
		elif command_input == "plot_data":
			process_plot_data_command(linear_regression)
		elif command_input == "train":
			process_train_command(linear_regression)
		elif command_input == "exit":
			break
		else:
			print("\nInvalid command.")

def show_banner():
	print(
		"\n"
		"////////////////////////////////////////////////////////////////////\n\n"
		"                        ft_linear_regression                        \n\n"
		"////////////////////////////////////////////////////////////////////"
		
	)

def show_welcome_message():
	print(
		"\nThis program uses a linear regression model to estimate the worth of a"
		"car given its mileage.\n"
		"The model is trained using pre-loaded data.\n"
		"The initial values for the intercept and slope parameter are zero.\n"
		"The generated graphs can be found within the ./graphs/ directory.")

def ask_for_commands():
	print("\nCommands available: 'estimate' 'plot_data' 'train' 'exit'.")

def process_estimate_command(linear_regression):
	while (True):
		try:
			estimation_input = input("\nEnter a non-negative mileage in km: ")
			if estimation_input == "exit":
				break
			estimation_input = int(estimation_input)
		except ValueError:
			display_not_a_number_warning()
			continue
		if estimation_input < 0:
			display_negative_estimation_input_warning()
		else:
			estimation = linear_regression.predict(estimation_input)
			display_estimation(estimation)

def process_plot_data_command(linear_regression):
	linear_regression.plot_data()

def display_not_a_number_warning():
	print("\nPlease provide a number.")

def display_negative_estimation_input_warning():
	print("\nThe mileage given was negative.")
	
def display_estimation(estimation):
	print(f"\nThe estimated value of the car is: {estimation}")
	if estimation < 0:
		print("\nWarning: The estimated value is negative.")

def process_train_command(linear_regression):
	display_training_start_status()
	linear_regression.train()
	display_training_end_status()
	display_training_results(linear_regression)

def display_training_start_status():
	print("\nTraining model...")

def display_training_end_status():
	print("\nTraining complete.\n")

def display_training_results(linear_regression):
	rmse, y_mean, mae, error_percentage = linear_regression.evaluate_model()
	print(f"Root Mean Squared Error: {round(rmse, 2)}")
	print(f"Mean value of price: {round(y_mean, 2)}")
	print(f"Mean Absolute Error: {round(mae, 2)}")
	print(f"Error Percentage: {round(error_percentage, 2)}%")
	linear_regression.plot_results()