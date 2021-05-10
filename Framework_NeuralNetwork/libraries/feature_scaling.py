class StandardScaler():
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        # the standard deviation can be 0, which provokes
        # devision-by-zero errors; let's avoid that:
        self.std[self.std == 0] = 0.00001

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X_scaled):
        return X_scaled * self.std + self.mean

    def weight_inverse_transform(self, weights):
        weights = weights.copy()
        weights[1:] /= self.std
        weights[0] -= (self.mean * weights[1:]).sum()
        return weights



class NormalScaler():
    # remember that we never want to normalize the bias column; `has_bias_column` is
    # a parameter we can set to avoid transforming the first column, to make our
    # lives easier later :-)
    def fit(self, X):
        self.min = X.min(axis=0).astype(float)
        self.max = X.max(axis=0).astype(float) - self.min
        self.max[self.max == 0] = 0.0001
    def transform(self, X):
        return (X - self.min) / self.max

    def inverse_transform(self, X_scaled):
        return X_scaled * self.max + self.min


