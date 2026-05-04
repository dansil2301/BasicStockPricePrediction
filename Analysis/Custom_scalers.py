import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def rolling_scale(X, window=100):
    X_scaled = np.zeros_like(X.values, dtype=float)
    scaler = StandardScaler()

    for i in range(len(X)):
        start = max(0, i - window)
        X_window = X.iloc[start:i+1]

        scaler.fit(X_window)
        X_scaled[i] = scaler.transform(X.iloc[i:i+1])[0]

    return X_scaled


def rolling_scale_with_history(X, history, window=100):
    X_scaled = np.zeros_like(X.values, dtype=float)
    scaler = StandardScaler()

    full_data = pd.concat([history, X])

    for i in range(len(X)):
        idx = len(history) + i
        start = max(0, idx - window)

        scaler.fit(full_data.iloc[start:idx+1])
        X_scaled[i] = scaler.transform(full_data.iloc[idx:idx+1])[0]

    return X_scaled
