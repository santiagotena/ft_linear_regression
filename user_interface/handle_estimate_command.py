from .interface_messages import *

def handle_estimate_command(linear_regression):
	while (True):
		estimation_input = input("\nEnter a non-negative mileage in km: ")
		if estimation_input == "exit":
			break
		try:
			estimation_input = int(estimation_input)
		except ValueError:
			display_not_a_number_warning()
			continue
		if estimation_input < 0:
			display_negative_estimation_input_warning()
		else:
			estimation = linear_regression.predict(estimation_input)
			display_estimation(estimation)
