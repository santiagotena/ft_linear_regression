import os
import matplotlib.pyplot as plt

def store_plot(plot_name):
	if not os.path.exists("graphs"):
		os.makedirs("graphs")
	plt.savefig("graphs/" + plot_name)
	plt.show()