import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from src.pipeline.pipeline import MLPipeline


def test_pipeline_classification():
    print("=" * 50)
    print("Testing Classification Pipeline")
    print("=" * 50)

    X, y = make_classification(n_samples=200, n_features=4, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
    df['target'] = y
    data_path = 'outputs/test_classification.csv'
    df.to_csv(data_path, index=False)

    pipeline = MLPipeline(
        raw_data_path=data_path,
        target_column='target',
        output_dir='outputs/test_classification_run',
        test_size=0.2,
        random_state=42
    )

    results, champion = pipeline.run()

    assert len(results) == 6 
    assert champion is not None
    assert 'model_name' in champion
    assert 'f1' in champion['metrics']
    print(f"\n✓ Classification pipeline passed. Champion: {champion['model_name']}")


def test_pipeline_regression():
    print("\n" + "=" * 50)
    print("Testing Regression Pipeline")
    print("=" * 50)

    X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
    df = pd.DataFrame(X, columns=['f1', 'f2', 'f3'])
    df['target'] = y
    data_path = 'outputs/test_regression.csv'
    df.to_csv(data_path, index=False)

    pipeline = MLPipeline(
        raw_data_path=data_path,
        target_column='target',
        output_dir='outputs/test_regression_run',
        test_size=0.2,
        random_state=42
    )

    results, champion = pipeline.run()

    assert len(results) == 6
    assert champion is not None
    assert champion['model_name'] in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'GradientBoosting']
    assert 'r2' in champion['metrics']
    print(f"\n✓ Regression pipeline passed. Champion: {champion['model_name']}")


if __name__ == '__main__':
    test_pipeline_classification()
    test_pipeline_regression()
    print("\n" + "=" * 50)
    print("All pipeline tests passed.")
    print("=" * 50)