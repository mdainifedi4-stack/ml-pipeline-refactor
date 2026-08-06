"""
Reusable metric calculation functions.
Replaces the repeated accuracy/precision/recall/f1 and RMSE/MAE/R2 blocks.
Returns plain dicts — no Flask, no global state.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score,
)


def compute_classification_metrics(y_true, y_pred, average='weighted'):
    """
    Calculate standard classification metrics.

    Replaces the ~6× copy-pasted metric block from KNN/DT/RF routes.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
        'classification_report': classification_report(y_true, y_pred, zero_division=0),
    }


def compute_regression_metrics(y_true, y_pred):
    """
    Calculate standard regression metrics.

    Replaces the 2× copy-pasted RMSE/MAE/R2 block from Linear Regression routes.
    """
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }


def compute_clustering_metrics(X, labels):
    """
    Calculate clustering quality metrics.

    For the manual clustering feature (kept in Flask app, but extracted here).
    """
    return {
        'silhouette': silhouette_score(X, labels),
    }