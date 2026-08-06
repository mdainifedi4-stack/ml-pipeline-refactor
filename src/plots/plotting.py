"""
Reusable plotting functions extracted from the Flask app.
All functions accept data + an output path, save the figure, and return the path.
No Flask dependencies. No global state.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; required before importing pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree


def _ensure_dir(file_path):
    """Create parent directories if they don't exist."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    return file_path


def plot_correlation(df, columns, target, output_path,
                     figsize=(10, 8), cmap='coolwarm', fmt='.2f',
                     title='Feature Correlation Matrix'):
    """
    Generate and save a correlation heatmap.

    Replaces the ~9× copy-pasted heatmap block from the Flask routes.
    """
    _ensure_dir(output_path)

    # Avoid duplicates if target is accidentally in columns
    cols = list(dict.fromkeys(list(columns) + [target]))
    corr = df[cols].corr()

    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=fmt, cmap=cmap)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()

    return Path(output_path)


def plot_confusion_matrix(y_true, y_pred, output_path,
                          figsize=(7, 5), dpi=100,
                          title='Confusion Matrix'):
    """
    Generate and save a confusion matrix heatmap.

    Replaces the ~7× copy-pasted confusion matrix block.
    """
    _ensure_dir(output_path)

    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))

    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=classes,
                yticklabels=classes,
                cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return Path(output_path)


def plot_feature_importance(model, feature_names, output_path,
                            figsize=(10, 6), title='Feature Importance'):
    """
    Horizontal bar chart of feature importances.
    For Random Forest and Decision Tree models.
    """
    _ensure_dir(output_path)

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    plt.figure(figsize=figsize)
    plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return Path(output_path)


def plot_decision_tree(tree_model, feature_names, class_names, output_path,
                       figsize=(50, 15), dpi=200):
    """
    Visualize a fitted Decision Tree.

    Replaces the 2× copy-pasted plot_tree block.
    """
    _ensure_dir(output_path)

    plt.figure(figsize=figsize, dpi=dpi)
    plot_tree(
        tree_model,
        filled=True,
        feature_names=feature_names,
        class_names=class_names.astype(str),
        rounded=True,
        fontsize=14
    )
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return Path(output_path)


def plot_actual_vs_predicted(y_true, y_pred, output_path,
                             figsize=(10, 6), title='Actual vs Predicted Values'):
    """
    Scatter plot with a perfect-prediction diagonal line.
    For regression models.
    """
    _ensure_dir(output_path)

    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', lw=2)
    plt.title(title)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return Path(output_path)


def plot_residuals(y_pred, residuals, output_path,
                   figsize=(10, 6), title='Residual Plot'):
    """
    Residuals vs predicted values scatter plot.
    For regression models.
    """
    _ensure_dir(output_path)

    plt.figure(figsize=figsize)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-', lw=2)
    plt.title(title)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return Path(output_path)