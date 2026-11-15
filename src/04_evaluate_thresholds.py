
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data/raw/application_train.csv"
MODEL_PATH = BASE_DIR / "models/xgboost_model.joblib"

# Load original data and saved model
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

features = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_CONTRACT_TYPE",
]

X = df[features].copy()
y = df["TARGET"].copy()

X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(
    365243, np.nan
)

# Reproduce the original test split
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]

print("=" * 65)
print("CREDITLENS AI | CLASSIFICATION THRESHOLD ANALYSIS")
print("=" * 65)

print(f"\nROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(
    "PR-AUC: "
    f"{average_precision_score(y_test, y_prob):.4f}"
)

print(
    "\nPredicted probability distribution "
    "(percentiles):"
)

for percentile in [0, 25, 50, 75, 90, 95, 99, 100]:
    value = np.percentile(y_prob, percentile)
    print(f"{percentile:3d}th percentile: {value:.4f}")

# Compare several research thresholds
thresholds = [
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
]

rows = []

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    rows.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "Predicted Positive": int(y_pred.sum()),
        "True Positive": int(tp),
        "False Positive": int(fp),
    })

results = pd.DataFrame(rows)

print("\nTHRESHOLD COMPARISON\n")

print(
    results.to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Precision": "{:.3f}".format,
            "Recall": "{:.3f}".format,
            "F1": "{:.3f}".format,
        },
    )
)

output_dir = BASE_DIR / "data/processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "xgboost_threshold_analysis.csv"

results.to_csv(output_path, index=False)

print(f"\nSaved results: {output_path}")
print("\nThreshold analysis completed!")
