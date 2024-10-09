from linear_regression import LinearRegression
from .interface_messages import *
from .handle_estimate_command import *
from .handle_plot_data_command import *
from .handle_train_command import *
from .utils import *

def launch_program():
	greet_user()
	linear_regression = LinearRegression()
	loop_interface(linear_regression)

def greet_user():
	display_banner()
	display_welcome_message()

def loop_interface(linear_regression):
	while(True):
		ask_for_commands()
		command_input = input("Enter a command: ")
		if command_input == "estimate":
			handle_estimate_command(linear_regression)
		elif command_input == "plot_data":
			handle_plot_data_command(linear_regression)
		elif command_input == "train":
			handle_train_command(linear_regression)
		elif command_input == "exit":
			break
		else:
			print("\nInvalid command.")
