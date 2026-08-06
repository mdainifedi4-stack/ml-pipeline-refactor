import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

from src.training.training import (
    split_and_scale,
    train_model,
    train_with_gridsearch,
    predict_and_evaluate,
)


def test_split_and_scale():
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
    y = pd.Series([0, 1, 0, 1, 0])
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, test_size=0.4, random_state=42)
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 2
    assert scaler is not None
    print("✓ split_and_scale passed")


def test_train_model():
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 2, 1]})
    y = pd.Series([0, 1, 0])
    model = KNeighborsClassifier(n_neighbors=1)
    trained = train_model(model, X, y)
    assert hasattr(trained, 'classes_')
    print("✓ train_model passed")


def test_train_with_gridsearch():
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6], 'b': [6, 5, 4, 3, 2, 1]})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = KNeighborsClassifier()
    param_grid = {'n_neighbors': [1, 2, 3]}
    best_model, best_params = train_with_gridsearch(model, param_grid, X, y, cv=3)
    assert best_model is not None
    assert 'n_neighbors' in best_params
    print("✓ train_with_gridsearch passed")


def test_predict_and_evaluate_classification():
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 2, 1]})
    y = pd.Series([0, 1, 0])
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X, y)
    y_pred, metrics = predict_and_evaluate(model, X, y, "classification")
    assert 'accuracy' in metrics
    assert y_pred is not None
    print("✓ predict_and_evaluate (classification) passed")


def test_predict_and_evaluate_regression():
    X = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [4, 3, 2, 1]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    model = LinearRegression()
    model.fit(X, y)
    y_pred, metrics = predict_and_evaluate(model, X, y, "regression")
    assert 'rmse' in metrics
    assert 'r2' in metrics
    print("✓ predict_and_evaluate (regression) passed")


if __name__ == '__main__':
    test_split_and_scale()
    test_train_model()
    test_train_with_gridsearch()
    test_predict_and_evaluate_classification()
    test_predict_and_evaluate_regression()
    print("\nAll training tests passed.")