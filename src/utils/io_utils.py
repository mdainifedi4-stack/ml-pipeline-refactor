"""
Utility functions for I/O operations.
Replaces the scattered timestamp, directory creation, and model serialization
code from the Flask routes.
"""
import time
from pathlib import Path

import joblib


def generate_timestamp():
    """
    Generate a unique integer timestamp string for filenames.

    Replaces: str(int(time.time()))
    """
    return str(int(time.time()))


def ensure_dir(file_path):
    """
    Create parent directories for a file path if they don't exist.

    Replaces the repeated os.makedirs(plot_dir, exist_ok=True) blocks.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    return Path(file_path)


def save_model_artifact(model, output_path):
    """
    Serialize a trained model to disk using joblib.

    Parameters
    ----------
    model : sklearn estimator
    output_path : str or Path

    Returns
    -------
    Path
    """
    ensure_dir(output_path)
    joblib.dump(model, output_path)
    return Path(output_path)


def load_model_artifact(path):
    """
    Load a serialized model from disk.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    object
        The loaded model.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)