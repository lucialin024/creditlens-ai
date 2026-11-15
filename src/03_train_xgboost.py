
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# 1. File paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data/raw/application_train.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load data
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

# Handle anomalous employment-duration values
X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(
    365243, np.nan
)


# 3. Same train/test split as baseline
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# 4. Preprocessing
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


# 5. XGBoost classifier
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


# 6. Train
print("=" * 60)
print("CREDITLENS AI | XGBOOST MODEL")
print("=" * 60)

print(f"Training observations: {len(X_train):,}")
print(f"Testing observations: {len(X_test):,}")

print("\nTraining XGBoost...")
model.fit(X_train, y_train)

print("Training complete!")


# 7. Evaluate
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("\nMODEL PERFORMANCE")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Average Precision (PR-AUC): {pr_auc:.4f}")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0,
))


# 8. Save the model
model_path = MODEL_DIR / "xgboost_model.joblib"

joblib.dump(model, model_path)

print(f"\nSaved model: {model_path}")
print("\nXGBoost training completed successfully!")

