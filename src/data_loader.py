"""
Data ingestion module for credit scoring pipeline.
Handles loading, initial validation, and train/test split.
"""

import pandas as pd
import os
from pathlib import Path


def load_data(filepath: str = "data/accepted_loan_data.csv") -> pd.DataFrame:
    """
    Load the LendingClub accepted loans dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Raw DataFrame.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Download from: https://www.kaggle.com/datasets/Barun/accepted-loans "
            "and place accepted_loan_data.csv in the data/ directory."
        )

    df = pd.read_csv(filepath, low_memory=False)
    print(f"[data_loader] Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that required columns exist in the dataset.

    Args:
        df: Input DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    required = ["loan_status", "int_rate", "annual_inc", "dti", "loan_amnt", "grade"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def split_data(
    df: pd.DataFrame,
    target_col: str = "loan_status",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split data into train and test sets, stratified on target.

    Args:
        df: Input DataFrame.
        target_col: Name of target column.
        test_size: Fraction for test split.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test tuples.
    """
    from sklearn.model_selection import train_test_split

    validate_schema(df)

    # Target: 1 = default, 0 = fully paid
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[data_loader] Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"[data_loader] Default rate — Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    print(df.head(3))