import numpy as np

def min_max_scale_fit(X):
    X = np.array(X, dtype=float)

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    scale = max_vals - min_vals
    scale[scale == 0] = 1.0

    return {
        "min": min_vals,
        "scale": scale
    }

def min_max_scale_transform(X, scaler):
    X = np.array(X, dtype=float)
    X_scaled = (X - scaler["min"]) / scaler["scale"]
    return X_scaled.tolist()  # convert back to list for scratch models
