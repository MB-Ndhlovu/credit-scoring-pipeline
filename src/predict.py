"""
Inference script for credit scoring model.
Load serialized model + preprocessors, predict on new loan applications.
"""

import joblib
import pandas as pd
import numpy as np


def load_artifacts(model_dir: str = "models"):
    """
    Load serialized model, encoder, and scaler.

    Args:
        model_dir: Directory containing saved artifacts.

    Returns:
        model, encoder dict, scaler, threshold.
    """
    model = joblib.load(f"{model_dir}/credit_model.pkl")
    encoder = joblib.load(f"{model_dir}/encoder.pkl")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")
    threshold = 0.5
    return model, encoder, scaler, threshold


def preprocess_new_application(df: pd.DataFrame, encoder, scaler):
    """
    Preprocess a new loan application the same way as training data.

    Args:
        df: Raw application DataFrame.
        encoder: Fitted encoder dict.
        scaler: Fitted scaler.

    Returns:
        Preprocessed features ready for prediction.
    """
    from features import create_derived_features, clean_missing, encode_categoricals

    df = create_derived_features(df)
    df = clean_missing(df)
    df, _ = encode_categoricals(df, encoder, fit=False)

    numeric_features = [
        "loan_amnt", "int_rate", "annual_inc", "dti", "delinq_2yrs",
        "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
        "total_acc", "balance_to_income", "loan_to_income",
        "credit_utilization", "debt_burden",
    ]
    available = [c for c in numeric_features if c in df.columns]
    df[available] = scaler.transform(df[available])

    return df


def predict(df: pd.DataFrame, model=None, encoder=None, scaler=None, threshold: float = None):
    """
    Predict default probability for one or more loan applications.

    Args:
        df: Raw application DataFrame.
        model: Trained model (auto-loads if None).
        encoder: Fitted encoder (auto-loads if None).
        scaler: Fitted scaler (auto-loads if None).
        threshold: Decision threshold (auto-loads from model if None).

    Returns:
        DataFrame with probability and decision for each application.
    """
    if model is None:
        model, encoder, scaler, default_thresh = load_artifacts()
        threshold = threshold or default_thresh

    X = preprocess_new_application(df, encoder, scaler)
    prob = model.predict_proba(X)[:, 1]
    decision = (prob >= threshold).astype(int)

    results = df[["loan_amnt", "int_rate", "annual_inc"]].copy()
    results["default_probability"] = prob
    results["model_decision"] = decision
    results["risk_label"] = results["decision"].apply(
        lambda d: "APPROVE" if d == 0 else "REVIEW/REJECT"
    )

    return results


def batch_predict(csv_path: str, output_path: str = "data/predictions.csv"):
    """
    Run predictions on a batch of applications from a CSV file.

    Args:
        csv_path: Path to input CSV with loan applications.
        output_path: Path to save results.

    Returns:
        Results DataFrame.
    """
    df = pd.read_csv(csv_path)
    results = predict(df)
    results.to_csv(output_path, index=False)
    print(f"[predict] Saved {len(results)} predictions to {output_path}")
    return results


if __name__ == "__main__":
    # Example: predict on a single synthetic application
    sample = pd.DataFrame({
        "loan_amnt": [15000],
        "int_rate": [14.5],
        "annual_inc": [55000],
        "dti": [22.0],
        "grade": ["B"],
        "revol_bal": [3000],
        "total_rev_hi_lim": [20000],
    })

    results = predict(sample)
    print(results.to_string(index=False))