
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ==================================================
# CREDITLENS AI | MODEL FEATURE IMPORTANCE
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR / "models" / "xgboost_model.joblib"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Load trained pipeline
# --------------------------------------------------

print("=" * 60)
print("CREDITLENS AI | XGBOOST FEATURE IMPORTANCE")
print("=" * 60)

model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]


# --------------------------------------------------
# 2. Retrieve transformed feature names
# --------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

importance_values = classifier.feature_importances_

if len(feature_names) != len(importance_values):
    raise ValueError(
        "Feature names and importance values do not match."
    )


# --------------------------------------------------
# 3. Create feature importance table
# --------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values,
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False,
).reset_index(drop=True)

print("\nTOP 20 TRANSFORMED FEATURES\n")

print(
    importance_df.head(20).to_string(
        index=False,
        formatters={
            "Importance": "{:.5f}".format,
        },
    )
)


# --------------------------------------------------
# 4. Aggregate one-hot encoded features
# --------------------------------------------------

# One-hot encoding creates multiple columns
# from a single original categorical feature.
# We combine these for a simpler dashboard view.

original_features = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_CONTRACT_TYPE",
]

def get_original_feature(transformed_feature):

    for original_feature in original_features:

        if (
            transformed_feature == original_feature
            or transformed_feature.endswith(
                "__" + original_feature
            )
            or transformed_feature.startswith(
                "categorical__" + original_feature + "_"
            )
        ):
            return original_feature

    raise ValueError(
        f"Unrecognized transformed feature: "
        f"{transformed_feature}"
    )


importance_df["Original_Feature"] = (
    importance_df["Feature"]
    .apply(get_original_feature)
)

grouped_importance = (
    importance_df
    .groupby("Original_Feature", as_index=False)
    ["Importance"]
    .sum()
    .sort_values(
        "Importance",
        ascending=False,
    )
    .reset_index(drop=True)
)


# --------------------------------------------------
# 5. Add readable feature labels
# --------------------------------------------------

feature_labels = {
    "AMT_INCOME_TOTAL": "Total Income",
    "AMT_CREDIT": "Credit Amount",
    "AMT_ANNUITY": "Annuity Amount",
    "AMT_GOODS_PRICE": "Goods Price",
    "DAYS_EMPLOYED": "Employment Duration",
    "NAME_INCOME_TYPE": "Income Type",
    "NAME_EDUCATION_TYPE": "Education Level",
    "NAME_CONTRACT_TYPE": "Loan Contract Type",
}

grouped_importance["Feature_Label"] = (
    grouped_importance["Original_Feature"]
    .map(feature_labels)
)

# Normalize for display as relative importance.

importance_sum = grouped_importance["Importance"].sum()

if importance_sum > 0:
    grouped_importance["Relative_Importance"] = (
        grouped_importance["Importance"] / importance_sum
    )
else:
    grouped_importance["Relative_Importance"] = 0.0


# --------------------------------------------------
# 6. Print grouped results
# --------------------------------------------------

print("\nGROUPED FEATURE IMPORTANCE\n")

print(
    grouped_importance[
        ["Feature_Label", "Relative_Importance"]
    ].to_string(
        index=False,
        formatters={
            "Relative_Importance": "{:.2%}".format,
        },
    )
)


# --------------------------------------------------
# 7. Save outputs
# --------------------------------------------------

detail_path = (
    OUTPUT_DIR / "xgboost_feature_importance_detail.csv"
)

grouped_path = (
    OUTPUT_DIR / "xgboost_feature_importance.csv"
)

importance_df.to_csv(
    detail_path,
    index=False,
)

grouped_importance.to_csv(
    grouped_path,
    index=False,
)

print(f"\nSaved: {detail_path}")
print(f"Saved: {grouped_path}")

print("\nFeature importance analysis completed!")