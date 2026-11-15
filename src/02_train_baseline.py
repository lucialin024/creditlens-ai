
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "application_train.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Start with a small set of applicant financial features.
# These are chosen for a simple, explainable baseline.

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

# Home Credit uses 365243 as an anomalous DAYS_EMPLOYED
# value. Treat it as missing rather than real employment.

X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(
    365243, np.nan
)

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

# --------------------------------------------------
# 3. Split into train and test datasets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("=" * 60)
print("CREDITLENS AI | LOGISTIC REGRESSION BASELINE")
print("=" * 60)

print(f"Training observations: {len(X_train):,}")
print(f"Testing observations:  {len(X_test):,}")
print(f"Training positive rate: {y_train.mean():.2%}")
print(f"Testing positive rate:  {y_test.mean():.2%}")

# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)

# --------------------------------------------------
# 5. Build model pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

# --------------------------------------------------
# 6. Train
# --------------------------------------------------

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training complete!")

# --------------------------------------------------
# 7. Evaluate
# --------------------------------------------------

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_prob)

pr_auc = average_precision_score(y_test, y_prob)

print("\nMODEL PERFORMANCE")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Average Precision (PR-AUC): {pr_auc:.4f}")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, zero_division=0))

print(
    "\nNote: class_weight='balanced' changes model fitting; "
    "raw predicted probabilities may require calibration."
)

# --------------------------------------------------
# 8. Save model
# --------------------------------------------------

model_path = MODEL_DIR / "logistic_regression.joblib"

joblib.dump(model, model_path)

print(f"\nSaved model: {model_path}")
print("\nBaseline modeling completed successfully!")