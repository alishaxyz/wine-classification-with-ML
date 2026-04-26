import math

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z):
        if z >= 0:
            e = math.exp(-z)
            return 1 / (1 + e)
        else:
            e = math.exp(z)
            return e / (1 + e)

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

        for _ in range(self.epochs):
            dw = [0.0] * n_features
            db = 0.0

            for i in range(n_samples):
                linear = self._dot(self.weights, X[i]) + self.bias
                y_pred = self._sigmoid(linear)

                error = y_pred - y[i]

                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error

            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n_samples)
            self.bias -= self.lr * (db / n_samples)

    def predict_proba(self, X):
        probs = []
        for x in X:
            linear = self._dot(self.weights, x) + self.bias
            probs.append(self._sigmoid(linear))
        return probs

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probs]
