"""Dimensionless SOH error metrics."""
import numpy as np


def evaluate(y_true, y_pred):
    y = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(pred):
        raise ValueError('nonempty truth and prediction arrays must have equal length')
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('nonfinite values in evaluation')
    diff = pred - y
    return {'MAE': float(np.mean(np.abs(diff))),
            'RMSE': float(np.sqrt(np.mean(diff**2))),
            'MAPE_percent': float(100 * np.mean(np.abs(diff) / np.maximum(np.abs(y), 1e-8))),
            'N': len(y)}
