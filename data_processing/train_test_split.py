import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
	X_len = X.shape[0]
	indices = np.arange(X_len)

	if random_state is not None:
		np.random.seed(random_state)
	np.random.shuffle(indices)
	
	test_elements = int(X_len * test_size)
	test_indices = indices[:test_elements]
	train_indices = indices[test_elements:]

	X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
	y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

	return X_train, X_test, y_train, y_test
