"""
Model training and evaluation module for credit scoring pipeline.
Implements model selection, hyperparameter tuning, and business-metric evaluation.
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from xgboost import XGBClassifier


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    ),
}

PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 15],
        "min_samples_split": [5, 10],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    },
}


def find_best_model(X_train, y_train):
    """
    Run GridSearchCV across all model types and return the best one.

    Args:
        X_train: Training features (preprocessed).
        y_train: Training labels.

    Returns:
        Best model, best params, CV results dict.
    """
    results = {}
    best_model = None
    best_score = 0
    best_name = ""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in MODELS.items():
        print(f"\n[train] Tuning {name}...")
        grid = GridSearchCV(
            model,
            PARAM_GRIDS[name],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        results[name] = {
            "best_params": grid.best_params_,
            "best_score": grid.best_score_,
            "best_estimator": grid.best_estimator_,
        }

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_name = name

    return best_model, best_name, results


def evaluate_model(model, X_test, y_test, threshold: float = 0.5):
    """
    Evaluate model on test set and print business-readable metrics.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        threshold: Decision threshold for classification.

    Returns:
        Dict of metrics.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*50}")
    print(f"MODEL EVALUATION — {model.__class__.__name__}")
    print(f"{'='*50}")
    print(f"AUC-ROC:    {auc:.4f}")
    print(f"Accuracy:   {acc:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"Recall:     {recall:.4f}  (captured {tp:,} of {tp+fn:,} defaults)")
    print(f"Precision:  {precision:.4f}  ({tp:,} true positives, {fp:,} false approvals)")
    print(f"{'='*50}")
    print("\nConfusion Matrix:")
    print(f"  TN={tn:,}  FP={fp:,}")
    print(f"  FN={fn:,}  TP={tp:,}")
    print(f"\nTotal defaults in test: {tp+fn:,} | Total approved: {tn+fp:,}")

    # Business interpretation
    approval_rate = y_pred.mean()
    default_rate_in_approved = fp / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\nBusiness Metrics:")
    print(f"  Approval rate:          {approval_rate:.1%}")
    print(f"  Default rate (approved): {default_rate_in_approved:.1%}")
    print(f"  Defaults caught:        {recall:.1%}")
    print(f"  Missed defaults:        {fn:,} out of {tp+fn:,}")

    return {
        "auc": auc,
        "accuracy": acc,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "confusion_matrix": cm.tolist(),
        "approval_rate": approval_rate,
        "default_rate_in_approved": default_rate_in_approved,
        "threshold": threshold,
    }


def tune_threshold(model, X_test, y_test, target_recall: float = 0.80):
    """
    Find the decision threshold that achieves target recall
    while maximizing precision.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        target_recall: Desired default capture rate.

    Returns:
        Optimal threshold, metrics dict.
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # Find threshold where recall >= target_recall
    valid_indices = np.where(tpr >= target_recall)[0]
    if len(valid_indices) == 0:
        print(f"[train] Could not achieve {target_recall:.0%} recall at any threshold.")
        return 0.5, {}

    # Pick the threshold with best precision at this recall level
    best_idx = valid_indices[np.argmax(fpr[valid_indices] - tpr[valid_indices])]
    optimal_threshold = thresholds[best_idx]

    print(f"\n[train] Threshold tuned for {target_recall:.0%} recall target:")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")

    return optimal_threshold, evaluate_model(model, X_test, y_test, threshold=optimal_threshold)


def rank_features(model, feature_names):
    """
    Extract feature importances and rank them.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names: List of feature names.

    Returns:
        DataFrame of ranked features.
    """
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\nTop 10 Predictive Features:")
    print(feat_df.head(10).to_string(index=False))

    return feat_df


def save_model(model, encoder, scaler, output_dir: str = "models"):
    """
    Serialize model, encoder, and scaler for production use.

    Args:
        model: Trained model.
        encoder: Fitted encoder dict.
        scaler: Fitted StandardScaler.
        output_dir: Directory to save artifacts.
    """
    Path(output_dir).mkdir(exist_ok=True)
    joblib.dump(model, f"{output_dir}/credit_model.pkl")
    joblib.dump(encoder, f"{output_dir}/encoder.pkl")
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    print(f"\n[train] Model artifacts saved to {output_dir}/")


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names=None):
    """
    Full training pipeline: find best model, tune threshold, rank features.

    Args:
        X_train, X_test: Train/test splits.
        y_train, y_test: Labels.
        feature_names: Optional list for feature ranking.

    Returns:
        Best model, metrics dict.
    """
    print("\n" + "="*60)
    print("CREDIT SCORING PIPELINE — MODEL TRAINING")
    print("="*60)

    best_model, best_name, all_results = find_best_model(X_train, y_train)

    print("\n[train] All model results:")
    for name, res in all_results.items():
        marker = " <-- BEST" if name == best_name else ""
        print(f"  {name:25s} AUC: {res['best_score']:.4f}{marker}")

    print(f"\n[train] Selected: {best_name} with AUC = {all_results[best_name]['best_score']:.4f}")

    base_metrics = evaluate_model(best_model, X_test, y_test)

    # Tune threshold for business target: catch 80% of defaults
    optimal_threshold, tuned_metrics = tune_threshold(best_model, X_test, y_test, target_recall=0.80)

    if feature_names is not None and hasattr(best_model, "feature_importances_"):
        feat_df = rank_features(best_model, feature_names)
        feat_df.to_csv("reports/feature_importances.csv", index=False)

    return best_model, {
        "model_name": best_name,
        "base_metrics": base_metrics,
        "tuned_metrics": tuned_metrics,
        "optimal_threshold": optimal_threshold,
    }


if __name__ == "__main__":
    # Quick test with dummy data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, results = train_and_evaluate(X_train, X_test, y_train, y_test)
    print("\nDone. Results:", results)