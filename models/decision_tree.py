class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def _gini(self, y):
        if len(y) == 0:
            return 0
        p1 = sum(y) / len(y)
        p0 = 1 - p1
        return 1 - p1 * p1 - p0 * p0

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gain = 0

        base_gini = self._gini(y)
        n_features = len(X[0])

        for f in range(n_features):
            values = sorted(set(x[f] for x in X))
            for t in values:
                left_y = [y[i] for i in range(len(X)) if X[i][f] <= t]
                right_y = [y[i] for i in range(len(X)) if X[i][f] > t]

                gini = (len(left_y) / len(y)) * self._gini(left_y) + \
                       (len(right_y) / len(y)) * self._gini(right_y)

                gain = base_gini - gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = t

        return best_feature, best_threshold

    def _build(self, X, y, depth):
        if depth == 0 or len(set(y)) == 1:
            return 1 if sum(y) >= len(y) / 2 else 0

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return 1 if sum(y) >= len(y) / 2 else 0

        left_X, left_y, right_X, right_y = [], [], [], []

        for i in range(len(X)):
            if X[i][feature] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build(left_X, left_y, depth - 1),
            "right": self._build(right_X, right_y, depth - 1),
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, self.max_depth)

    def _predict_one(self, x, node):
        if not isinstance(node, dict):
            return node

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]
