import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from src.metrics.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_clustering_metrics,
)


def test_classification_metrics():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'classification_report' in metrics
    assert 0.0 <= metrics['accuracy'] <= 1.0
    print("✓ compute_classification_metrics passed")


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert metrics['rmse'] > 0
    print("✓ compute_regression_metrics passed")


def test_clustering_metrics():
    X = np.array([[1, 2], [1, 3], [2, 2], [8, 8], [8, 9], [9, 8]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    metrics = compute_clustering_metrics(X, labels)
    assert 'silhouette' in metrics
    assert -1.0 <= metrics['silhouette'] <= 1.0
    print("✓ compute_clustering_metrics passed")


if __name__ == '__main__':
    test_classification_metrics()
    test_regression_metrics()
    test_clustering_metrics()
    print("\nAll metrics tests passed.")