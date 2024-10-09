import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=42):
	sample_number = X.shape[0]
	indices = np.arange(sample_number)

	np.random.seed(random_state)
	np.random.shuffle(indices)

	test_size = int(sample_number * test_size)

	test_indices = indices[:test_size]
	train_indices = indices[test_size:]

	X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
	y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

	return X_train, X_test, y_train, y_test