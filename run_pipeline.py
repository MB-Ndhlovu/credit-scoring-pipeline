"""
Credit Scoring Pipeline — run script
Generates synthetic credit data and executes the full pipeline.
Output: model comparison report and saved artifacts.
"""

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    accuracy_score
)
import joblib
import os
from datetime import datetime

np.random.seed(42)

print("=" * 60)
print("CREDIT SCORING PIPELINE — FULL EXECUTION")
print("=" * 60)

# 1. Generate synthetic credit data
print("\n[1/6] Generating synthetic credit dataset...")
n = 5000
data = pd.DataFrame({
    "credit_score": np.random.randint(300, 850, n),
    "annual_income": np.random.randint(20000, 250000, n),
    "debt_to_income": np.round(np.random.uniform(0.05, 0.55, n), 4),
    "employment_years": np.random.randint(0, 40, n),
    "loan_amount": np.random.randint(1000, 50000, n),
    "interest_rate": np.round(np.random.uniform(0.04, 0.28, n), 4),
    "loan_purpose": np.random.choice(["debt_consolidation", "home_improvement", "major_purchase", "business", "other"], n),
    "home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n),
    "verified_income": np.random.choice([0, 1], n, p=[0.35, 0.65]),
    "num_credit_lines": np.random.randint(1, 15, n),
    "delinquency_2yrs": np.random.randint(0, 5, n),
})
# Target: credit_score_band (0-3), income_band (0-3), employment_band (0-2)
# Score bands with weighted risk
score_band = np.where(data["credit_score"] < 550, 3,
               np.where(data["credit_score"] < 650, 2,
               np.where(data["credit_score"] < 750, 1, 0)))
income_band = np.where(data["annual_income"] < 35000, 3,
               np.where(data["annual_income"] < 65000, 2,
               np.where(data["annual_income"] < 120000, 1, 0)))
emp_band = np.where(data["employment_years"] < 1, 3,
             np.where(data["employment_years"] < 3, 2, 0))
delq_penalty = data["delinquency_2yrs"] * 2
dti_penalty = (data["debt_to_income"] > 0.40).astype(int) * 3

# Linear risk score (lower = more likely to default)
risk_score = score_band + income_band + emp_band + delq_penalty + dti_penalty
# Convert to probability via sigmoid
prob_default = 1 / (1 + np.exp(-0.35 * (risk_score - 9)))
data["default"] = (np.random.random(n) < prob_default).astype(int)
# Enforce ~22% default rate
while data["default"].mean() < 0.20:
    mask = (data["default"] == 0) & (np.random.random(n) < 0.10)
    data.loc[mask, "default"] = 1
while data["default"].mean() > 0.25:
    mask = (data["default"] == 1) & (np.random.random(n) < 0.10)
    data.loc[mask, "default"] = 0

print(f"   Dataset shape: {data.shape}")
print(f"   Default rate:  {data['default'].mean():.1%}")

# 2. Train/test split
print("\n[2/6] Train/test split...")
X = data.drop("default", axis=1)
y = data["default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 3. Preprocessing
print("\n[3/6] Feature engineering + encoding...")
cat_cols = ["loan_purpose", "home_ownership"]
X_train_enc = pd.get_dummies(X_train, columns=cat_cols)
X_test_enc = pd.get_dummies(X_test, columns=cat_cols)
# Align columns
for col in X_train_enc.columns:
    if col not in X_test_enc.columns:
        X_test_enc[col] = 0
X_test_enc = X_test_enc[X_train_enc.columns]

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_enc)
X_test_sc = scaler.transform(X_test_enc)

print(f"   Features after encoding: {X_train_enc.shape[1]}")

# 4. Model training
print("\n[4/6] Training models...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}
results = {}
for name, model in models.items():
    print(f"   Training {name}...")
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"model": model, "auc": auc, "f1": f1, "accuracy": acc, "y_prob": y_prob}
    print(f"   → AUC: {auc:.4f} | F1: {f1:.4f} | Accuracy: {acc:.4f}")

# 5. Best model selection
print("\n[5/6] Model comparison...")
best_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]
best_auc = results[best_name]["auc"]
print(f"\n   ★ BEST MODEL: {best_name} (AUC: {best_auc:.4f})")

print("\n   Full Classification Report — Best Model:")
y_pred_best = best_model.predict(X_test_sc)
print(classification_report(y_test, y_pred_best, target_names=["No Default", "Default"]))

print("\n   Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(f"   TN: {cm[0][0]:4d}  FP: {cm[0][1]:4d}")
print(f"   FN: {cm[1][0]:4d}  TP: {cm[1][1]:4d}")

# Business thresholds
print("\n   Business Threshold Analysis (at 0.5 cutoff):")
thresholds = [0.3, 0.4, 0.5, 0.6]
for t in thresholds:
    preds = (results[best_name]["y_prob"] >= t).astype(int)
    approve_rate = (preds == 0).mean()
    default_capture = (preds[y_test == 1] == 1).sum() / y_test.sum()
    print(f"   Threshold {t:.1f} → Approval rate: {approve_rate:.1%} | Default capture: {default_capture:.1%}")

# 6. Save artifacts
print("\n[6/6] Saving artifacts...")
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(best_model, "models/credit_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(list(X_train_enc.columns), "models/feature_names.pkl")

summary = {
    "best_model": best_name,
    "auc": round(best_auc, 4),
    "f1": round(results[best_name]["f1"], 4),
    "accuracy": round(results[best_name]["accuracy"], 4),
    "n_features": int(X_train_enc.shape[1]),
    "dataset_size": n,
    "default_rate": round(data["default"].mean(), 4),
    "run_date": datetime.now().isoformat(),
}
with open("reports/model_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("EXECUTION COMPLETE")
print("=" * 60)
print(f"\nBest model : {summary['best_model']}")
print(f"AUC        : {summary['auc']}")
print(f"F1 Score   : {summary['f1']}")
print(f"Accuracy   : {summary['accuracy']}")
print(f"Features   : {summary['n_features']}")
print(f"Artifacts  : models/credit_model.pkl")
print(f"            : models/scaler.pkl")
print(f"            : models/feature_names.pkl")
print(f"            : reports/model_results.json")