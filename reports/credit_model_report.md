# Credit Scoring Model — Business Report

**Model:** XGBoost Binary Classifier
**Purpose:** Predict loan default probability for credit underwriting
**Dataset:** LendingClub Accepted Loans (historical approved applications)
**Date:** May 2026

---

## Executive Summary

This model was built to support credit underwriting decisions. It predicts the probability that a borrower will default on a loan within the agreed repayment period.

**Business objective:** Maximize approval of viable borrowers while keeping the default rate below an acceptable risk threshold. This directly supports financial inclusion goals — ensuring creditworthy customers aren't unnecessarily excluded.

---

## Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| AUC-ROC | 0.79 | Good discriminative ability — model separates defaults from non-defaults effectively |
| Accuracy | 85% | Overall correct predictions |
| Recall | 53% | Model captures 53% of actual defaults |
| Precision | 61% | Of all predicted defaults, 61% are true defaults |
| F1 Score | 0.61 | Balanced performance on imbalanced dataset |
| Approval Rate | ~80% | Proportion of applications approved at default threshold |

**Threshold strategy:** Default threshold = 0.50. Tuned threshold = 0.41 to achieve 80% recall target (catching 80% of all defaults). Lowering threshold increases default detection but also increases false positives — the tradeoff is manageable with a secondary manual review queue.

---

## Top Predictive Features

1. **Interest Rate (int_rate)** — Higher rates correlate strongly with default. Primary signal.
2. **Debt-to-Income Ratio (dti)** — Existing debt burden is a top risk indicator.
3. **Annual Income (annual_inc)** — Higher income → lower default probability.
4. **Loan Amount (loan_amnt)** — Larger loans carry inherently more risk.
5. **Inquiry Last 6 Months (inq_last_6mths)** — Multiple recent inquiries signal financial stress.

---

## Business Impact

### What the model enables:

- **Faster decisions:** Automated approval for low-risk applications. Manual review reduced by ~45%.
- **Better risk selection:** Catch more defaults before origination — reduces net charge-off rate.
- **Consistent underwriting:** Eliminates human inconsistency in similar application evaluation.
- **Financial inclusion support:** Model approves creditworthy borrowers who might be incorrectly rejected by naive rule-based systems.

### Decisions the model supports:

1. **Approve:** default_probability < 0.35 → pass through automatically
2. **Review:** 0.35 ≤ default_probability < 0.55 → manual assessment
3. **Reject:** default_probability ≥ 0.55 → decline with explanation

---

## Model Limitations

- Trained on historical LendingClub data — may not generalize to all market segments, especially African markets with different credit behavior patterns
- Sensitive to distribution shift over time (economic conditions change) — monitoring required
- Missing features in source data: rent vs own, utility payments, mobile money history — would improve African market applicability
- Cannot capture employment stability, business ownership, or remittance income from informal economy

---

## Recommendations for African Market Deployment

1. **Retrain on local data** — South African/African fintech datasets would improve model relevance
2. **Add alternative credit signals** — mobile money repayment history, subscription payments, airtime purchase patterns
3. **Segment by market** — country-specific models perform better than single regional model
4. **Set conservative initial thresholds** — monitor for first 90 days before tightening approval criteria
5. **Bias testing** — validate across demographic groups to ensure fair access

---

## Technical Details

- **Algorithms evaluated:** Logistic Regression, Random Forest, XGBoost
- **Hyperparameter tuning:** GridSearchCV with 5-fold stratified cross-validation
- **Evaluation metric:** AUC-ROC (primary), F1 (secondary for class imbalance)
- **Class imbalance handling:** XGBoost with scale_pos_weight, class-weighted Random Forest
- **Production deployment:** FastAPI REST endpoint, model artifacts serialized via joblib
- **Monitoring:** Prediction distribution tracking, AUC drift detection

---

*Built by Malibongwe Ndhlovu — BSc Mathematical Sciences, Actuarial Science focus*