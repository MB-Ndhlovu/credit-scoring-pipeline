"""
Feature engineering module for credit scoring pipeline.
Handles preprocessing, encoding, missing value imputation, and feature creation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mths_since_last_major_derog",
    "acc_now_delinq",
    "total_rev_hi_lim",
]

CATEGORICAL_FEATURES = [
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
]

DERIVED_FEATURES = [
    "balance_to_income",      # revolving balance / annual income
    "loan_to_income",         # loan amount / annual income
    "credit_utilization",     # revolving balance / total revolving credit limit
    "debt_burden",            # dti * loan_amnt approximation
]


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features from raw columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with new derived columns.
    """
    df = df.copy()

    df["balance_to_income"] = df["revol_bal"] / (df["annual_inc"] + 1)
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)
    df["credit_utilization"] = df["revol_bal"] / (df["total_rev_hi_lim"] + 1)
    df["debt_burden"] = df["dti"] * df["loan_amnt"] / 1000

    return df


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values with appropriate strategies.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with imputed values.
    """
    df = df.copy()

    # Numeric: median imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Categorical: fill with "Unknown"
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df


def encode_categoricals(df: pd.DataFrame, encoder=None, fit: bool = True):
    """
    Label-encode categorical features.

    Args:
        df: Input DataFrame.
        encoder: Optional pre-fitted LabelEncoder.
        fit: If True, fit on data; if False, use provided encoder.

    Returns:
        DataFrame with encoded categoricals, fitted encoder.
    """
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)

        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoder.get(col)
            if le is None:
                continue
            # Handle unseen categories
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    return df, encoders if fit else encoder


def preprocess(df: pd.DataFrame, encoder=None, scaler=None, fit: bool = True):
    """
    Full preprocessing pipeline: derived features → clean → encode → scale.

    Args:
        df: Input DataFrame.
        encoder: Optional pre-fitted encoder dict.
        scaler: Optional pre-fitted StandardScaler.
        fit: If True, fit transformers; if False, transform only.

    Returns:
        Preprocessed DataFrame, encoder dict, scaler.
    """
    df = create_derived_features(df)
    df = clean_missing(df)
    df, encoder = encode_categoricals(df, encoder, fit)

    all_numeric = NUMERIC_FEATURES + DERIVED_FEATURES
    available_numeric = [c for c in all_numeric if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[available_numeric] = scaler.fit_transform(df[available_numeric])
    else:
        df[available_numeric] = scaler.transform(df[available_numeric])

    return df, encoder, scaler


if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    df_proc, _, _ = preprocess(df)
    print(df_proc.shape)