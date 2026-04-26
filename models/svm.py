class SVM:
    def __init__(self, lr=0.01, lambda_param=0.01, epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def _dot(self, a, b):
        s = 0.0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])

        self.weights = [0.0] * n_features
        self.bias = 0.0

        y_ = [1 if label == 1 else -1 for label in y]

        for _ in range(self.epochs):
            for i in range(n_samples):
                condition = y_[i] * (self._dot(self.weights, X[i]) + self.bias) >= 1

                if condition:
                    for j in range(n_features):
                        self.weights[j] -= self.lr * (2 * self.lambda_param * self.weights[j])
                else:
                    for j in range(n_features):
                        self.weights[j] -= self.lr * (2 * self.lambda_param * self.weights[j] - y_[i] * X[i][j])
                    self.bias -= self.lr * (-y_[i])

    def decision_function(self, X):
        scores = []
        for x in X:
            s = self._dot(self.weights, x) + self.bias
            scores.append(s)
        return scores

    def predict(self, X):
        preds = []
        for x in X:
            approx = self._dot(self.weights, x) + self.bias
            preds.append(1 if approx >= 0 else 0)
        return preds
