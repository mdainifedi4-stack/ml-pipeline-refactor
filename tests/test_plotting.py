import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

from src.plots.plotting import (
    plot_correlation,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_decision_tree,
    plot_actual_vs_predicted,
    plot_residuals,
)


def test_plot_correlation():
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [5, 4, 3, 2, 1],
        'target': [0, 1, 0, 1, 0]
    })
    path = plot_correlation(df, ['a', 'b'], 'target', 'outputs/test_corr.png')
    assert path.exists()
    print("✓ plot_correlation passed")


def test_plot_confusion_matrix():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    path = plot_confusion_matrix(y_true, y_pred, 'outputs/test_cm.png')
    assert path.exists()
    print("✓ plot_confusion_matrix passed")


def test_plot_feature_importance():
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 2, 1]})
    y = np.array([0, 1, 0])
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    path = plot_feature_importance(model, X.columns, 'outputs/test_fi.png')
    assert path.exists()
    print("✓ plot_feature_importance passed")


def test_plot_decision_tree():
    X = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [4, 3, 2, 1]})
    y = np.array([0, 1, 0, 1])
    model = DecisionTreeClassifier(max_depth=2, random_state=42)
    model.fit(X, y)
    path = plot_decision_tree(model, X.columns, model.classes_, 'outputs/test_dt.png')
    assert path.exists()
    print("✓ plot_decision_tree passed")


def test_plot_actual_vs_predicted():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    path = plot_actual_vs_predicted(y_true, y_pred, 'outputs/test_avp.png')
    assert path.exists()
    print("✓ plot_actual_vs_predicted passed")


def test_plot_residuals():
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    residuals = np.array([-0.1, 0.1, -0.2, 0.2, -0.1])
    path = plot_residuals(y_pred, residuals, 'outputs/test_resid.png')
    assert path.exists()
    print("✓ plot_residuals passed")


if __name__ == '__main__':
    test_plot_correlation()
    test_plot_confusion_matrix()
    test_plot_feature_importance()
    test_plot_decision_tree()
    test_plot_actual_vs_predicted()
    test_plot_residuals()
    print("\nAll plotting tests passed.")