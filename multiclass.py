class OneVsRest:
    def __init__(self, base_model_cls, **model_kwargs):
        self.base_model_cls = base_model_cls
        self.model_kwargs = model_kwargs
        self.models = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = sorted(set(y))
        self.models = {}

        for cls in self.classes:
            y_binary = [1 if label == cls else 0 for label in y]
            model = self.base_model_cls(**self.model_kwargs)
            model.fit(X, y_binary)
            self.models[cls] = model

    def predict(self, X):
        preds = []

        for x in X:
            best_class = None
            best_score = None

            for cls, model in self.models.items():
                if hasattr(model, "predict_proba"):
                    score = model.predict_proba([x])[0]
                elif hasattr(model, "decision_function"):
                    score = model.decision_function([x])[0]
                else:
                    score = model.predict([x])[0]

                if best_score is None or score > best_score:
                    best_score = score
                    best_class = cls

            preds.append(best_class)

        return preds
