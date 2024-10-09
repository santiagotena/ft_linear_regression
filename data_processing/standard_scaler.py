class StandardScaler:
	def fit(self, X):
		self.mean = X.mean(axis=0)
		self.std = X.std(axis=0, ddof=0)
		return self

	def transform(self, X):
		return (X - self.mean) / self.std

	def fit_transform(self, X):
		return self.fit(X).transform(X)
