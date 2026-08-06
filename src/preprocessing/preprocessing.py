"""
Data loading, cleaning, and problem-type detection.
Replaces the global-state-dependent preprocess_data() and the CSV/Excel loading
scattered across Flask routes.
"""
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(file_path):
    """
    Load a dataset from CSV or Excel.

    Parameters
    ----------
    file_path : str or Path

    Returns
    -------
    pd.DataFrame
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ('.xls', '.xlsx', '.xlsm'):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .csv or .xlsx")


def clean_data(df):
    """
    Clean a raw DataFrame:
    - Impute missing values (median for numeric, mode for categorical/datetime)
    - Label-encode categorical columns
    - Return both the cleaned DataFrame and a report dict.

    Replaces the preprocess_data() function from the Flask app.
    """
    processed_df = df.copy()
    report = {
        'missing_values': {},
        'categorical_encoded': [],
        'numerical_scaled': []
    }

    for col in processed_df.columns:
        missing = processed_df[col].isnull().sum()
        report['missing_values'][col] = missing

        if missing > 0:
            if pd.api.types.is_numeric_dtype(processed_df[col]):
                median_v = processed_df[col].median()
                processed_df[col] = processed_df[col].fillna(median_v)
                report['missing_values'][col] = f"Filled with median: {median_v:.2f}"
            elif pd.api.types.is_datetime64_any_dtype(processed_df[col]):
                mode_v = processed_df[col].mode()[0]
                processed_df[col] = processed_df[col].fillna(mode_v)
                report['missing_values'][col] = f"Filled with mode: {mode_v}"
            else:
                mode_v = processed_df[col].mode()[0]
                processed_df[col] = processed_df[col].fillna(mode_v)
                report['missing_values'][col] = f"Filled with mode: {mode_v}"

    for col in processed_df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col])
        report['categorical_encoded'].append({
            'column': col,
            'n_categories': len(le.classes_),
            'mapping': dict(zip(le.classes_, range(len(le.classes_))))
        })

    return processed_df, report


def detect_problem_type(y):
    """
    Auto-detect whether a target column represents classification or regression.

    Heuristic:
    - If dtype is object/category -> classification
    - If numeric and <= 10 unique values -> classification
    - Otherwise -> regression

    Parameters
    ----------
    y : pd.Series
        The target column.

    Returns
    -------
    str
        "classification" or "regression"
    """
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return "classification"

    if pd.api.types.is_numeric_dtype(y):
        n_unique = y.nunique()
        # Threshold: <= 10 unique values in a numeric column -> treat as classification
        if n_unique <= 10:
            return "classification"
        return "regression"

    # Fallback for datetime or other types
    return "classification"


def save_processed(df, output_path):
    """
    Save a processed DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
    output_path : str or Path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return Path(output_path)