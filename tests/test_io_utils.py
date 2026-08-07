import os
from pathlib import Path

from sklearn.linear_model import LinearRegression
import numpy as np

from src.utils.io_utils import (
    generate_timestamp,
    ensure_dir,
    save_model_artifact,
    load_model_artifact,
)


def test_generate_timestamp():
    ts1 = generate_timestamp()
    ts2 = generate_timestamp()
    assert ts1.isdigit()
    assert ts2.isdigit()
    assert int(ts2) >= int(ts1)
    print("✓ generate_timestamp passed")


def test_ensure_dir():
    path = "outputs/deep/nested/test_file.txt"
    result = ensure_dir(path)
    assert result.parent.exists()
    print("✓ ensure_dir passed")


def test_save_and_load_model():
    model = LinearRegression()
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    model.fit(X, y)

    path = "outputs/test_model.pkl"
    save_model_artifact(model, path)
    assert os.path.exists(path)

    loaded = load_model_artifact(path)
    prediction = loaded.predict(np.array([[4]]))
    assert np.isclose(prediction[0], 4.0)
    print("✓ save_model_artifact + load_model_artifact passed")


if __name__ == '__main__':
    test_generate_timestamp()
    test_ensure_dir()
    test_save_and_load_model()
    print("\nAll I/O utils tests passed.")