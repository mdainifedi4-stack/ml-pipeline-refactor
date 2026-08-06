import pandas as pd
import numpy as np
import os

from src.preprocessing.preprocessing import (
    load_data,
    clean_data,
    detect_problem_type,
    save_processed,
)


def test_load_data_csv():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    path = 'outputs/test_raw.csv'
    df.to_csv(path, index=False)
    loaded = load_data(path)
    assert loaded.shape == (3, 2)
    print("✓ load_data (CSV) passed")


def test_clean_data():
    df = pd.DataFrame({
        'num': [1.0, 2.0, np.nan, 4.0],
        'cat': ['a', 'b', 'a', np.nan],
        'target': [0, 1, 0, 1]
    })
    cleaned, report = clean_data(df)
    assert cleaned['num'].isnull().sum() == 0
    assert cleaned['cat'].isnull().sum() == 0
    assert 'missing_values' in report
    assert len(report['categorical_encoded']) >= 1
    print("✓ clean_data passed")


def test_detect_problem_type_classification():
    y = pd.Series([0, 1, 0, 1, 0])
    assert detect_problem_type(y) == "classification"
    print("✓ detect_problem_type (classification) passed")


def test_detect_problem_type_regression():
    y = pd.Series([1.5, 2.3, 4.1, 5.0, 6.2, 7.1, 8.3, 9.0, 10.5, 11.2, 12.0])
    assert detect_problem_type(y) == "regression"
    print("✓ detect_problem_type (regression) passed")


def test_detect_problem_type_categorical():
    y = pd.Series(['cat', 'dog', 'cat', 'dog'])
    assert detect_problem_type(y) == "classification"
    print("✓ detect_problem_type (categorical) passed")


def test_save_processed():
    df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
    path = 'outputs/test_processed.csv'
    save_processed(df, path)
    assert os.path.exists(path)
    loaded = pd.read_csv(path)
    assert loaded.shape == (2, 2)
    print("✓ save_processed passed")


if __name__ == '__main__':
    test_load_data_csv()
    test_clean_data()
    test_detect_problem_type_classification()
    test_detect_problem_type_regression()
    test_detect_problem_type_categorical()
    test_save_processed()
    print("\nAll preprocessing tests passed.")