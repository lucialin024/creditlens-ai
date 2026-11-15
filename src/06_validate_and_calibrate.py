from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# ==================================================
# CREDITLENS AI
# Model Validation & Probability Calibration
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data/raw/application_train.csv"

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data/processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

print("=" * 65)
print("CREDITLENS AI | MODEL VALIDATION & CALIBRATION")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

numeric_features = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
]

categorical_features = [
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_CONTRACT_TYPE",
]

features = numeric_features + categorical_features

X = df[features].copy()
y = df["TARGET"].copy()

# Correct special missing-value encoding.
X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(
    365243,
    np.nan,
)


# --------------------------------------------------
# 2. Training / validation / test split
# --------------------------------------------------

# First preserve the same 20% evaluation partition
# used in our previous modeling scripts.
#
# This partition has already been explored, so it
# is not a completely untouched final holdout.

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Split development data into:
# 80% for model training
# 20% for probability calibration
#
# Overall split:
# 64% training, 16% validation, 20% testing.

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.20,
    random_state=42,
    stratify=y_dev,
)

print("\nDATA SPLIT")
print(f"Training:   {len(X_train):,}")
print(f"Validation: {len(X_val):,}")
print(f"Testing:    {len(X_test):,}")

print(f"\nTraining positive rate: {y_train.mean():.2%}")
print(f"Validation positive rate: {y_val.mean():.2%}")
print(f"Testing positive rate: {y_test.mean():.2%}")


# --------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])


# --------------------------------------------------
# 4. XGBoost model
# --------------------------------------------------

classifier = XGBClassifier(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=4,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier),
])


# --------------------------------------------------
# 5. Train on training data only
# --------------------------------------------------

print("\nTraining XGBoost...")

model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# 6. Fit calibration on validation data
# --------------------------------------------------

print("\nFitting isotonic probability calibration...")

val_raw = model.predict_proba(X_val)[:, 1]

calibrator = IsotonicRegression(
    y_min=0.0,
    y_max=1.0,
    out_of_bounds="clip",
)

calibrator.fit(val_raw, y_val)

print("Calibration completed.")


# --------------------------------------------------
# 7. Evaluate on test data
# --------------------------------------------------

test_raw = model.predict_proba(X_test)[:, 1]

test_calibrated = calibrator.predict(test_raw)

# Clip only for numerical stability in log loss.
epsilon = 1e-7

def evaluate_predictions(name, probabilities):

    safe_probs = np.clip(
        probabilities,
        epsilon,
        1 - epsilon,
    )

    return {
        "Model": name,
        "ROC_AUC": roc_auc_score(
            y_test,
            probabilities,
        ),
        "PR_AUC": average_precision_score(
            y_test,
            probabilities,
        ),
        "Brier_Score": brier_score_loss(
            y_test,
            probabilities,
        ),
        "Log_Loss": log_loss(
            y_test,
            safe_probs,
        ),
    }


metrics = pd.DataFrame([
    evaluate_predictions(
        "XGBoost - Raw",
        test_raw,
    ),
    evaluate_predictions(
        "XGBoost - Calibrated",
        test_calibrated,
    ),
])

print("\nTEST-SET MODEL PERFORMANCE\n")

print(
    metrics.round(5).to_string(index=False)
)

print(
    "\nLower Brier Score and Log Loss are better."
)

print(
    "Higher ROC-AUC and PR-AUC are better."
)


# --------------------------------------------------
# 8. Calibration curve data
# --------------------------------------------------

def make_calibration_bins(
    probabilities,
    labels,
    model_name,
    n_bins=10,
):
    temp = pd.DataFrame({
        "Predicted": probabilities,
        "Observed": np.asarray(labels),
    })

    # Equal-frequency bins usually give more stable
    # empirical rates than fixed-width bins.

    temp["Bin"] = pd.qcut(
        temp["Predicted"],
        q=n_bins,
        duplicates="drop",
    )

    summary = (
        temp.groupby("Bin", observed=True)
        .agg(
            Mean_Predicted=("Predicted", "mean"),
            Observed_Rate=("Observed", "mean"),
            Count=("Observed", "size"),
        )
        .reset_index(drop=True)
    )

    summary["Model"] = model_name

    return summary


raw_bins = make_calibration_bins(
    test_raw,
    y_test,
    "XGBoost - Raw",
)

calibrated_bins = make_calibration_bins(
    test_calibrated,
    y_test,
    "XGBoost - Calibrated",
)

calibration_df = pd.concat(
    [raw_bins, calibrated_bins],
    ignore_index=True,
)

print("\nCALIBRATION CURVE DATA\n")

print(
    calibration_df.round(4).to_string(index=False)
)


# --------------------------------------------------
# 9. Save model and calibration artifacts
# --------------------------------------------------

model_path = (
    MODEL_DIR / "xgboost_calibration_base.joblib"
)

calibrator_path = (
    MODEL_DIR / "xgboost_isotonic_calibrator.joblib"
)

metrics_path = (
    OUTPUT_DIR / "calibration_metrics.csv"
)

curve_path = (
    OUTPUT_DIR / "calibration_curve.csv"
)

joblib.dump(model, model_path)
joblib.dump(calibrator, calibrator_path)

metrics.to_csv(
    metrics_path,
    index=False,
)

calibration_df.to_csv(
    curve_path,
    index=False,
)

print("\nSAVED OUTPUTS")
print(f"Model:       {model_path}")
print(f"Calibrator:  {calibrator_path}")
print(f"Metrics:     {metrics_path}")
print(f"Curve data:  {curve_path}")

print("\nModel validation and calibration completed!")
