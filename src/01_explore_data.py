
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Locate dataset
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "application_train.csv"

print("=" * 60)
print("CREDITLENS AI | CREDIT DEFAULT DATA EXPLORATION")
print("=" * 60)

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\n1. DATASET OVERVIEW")
print(f"Total applications: {len(df):,}")
print(f"Total columns: {df.shape[1]:,}")

print("\nFirst five rows:")
print(df.head().to_string())

# --------------------------------------------------
# 3. Understand target variable
# --------------------------------------------------

# TARGET = 0: No observed repayment difficulty
# TARGET = 1: Observed repayment difficulty

print("\n2. TARGET DISTRIBUTION")

target_counts = df["TARGET"].value_counts().sort_index()
target_rates = df["TARGET"].value_counts(normalize=True).sort_index() * 100

for target in target_counts.index:
    label = (
        "Repayment difficulty"
        if target == 1
        else "No repayment difficulty"
    )
    print(
        f"{label}: {target_counts[target]:,} "
        f"({target_rates[target]:.2f}%)"
    )

# --------------------------------------------------
# 4. Missing data
# --------------------------------------------------

print("\n3. TOP 15 COLUMNS WITH MISSING VALUES")

missing_pct = df.isna().mean().mul(100).sort_values(ascending=False)

print(
    missing_pct.head(15)
    .round(2)
    .to_string()
)

# --------------------------------------------------
# 5. Financial features
# --------------------------------------------------

print("\n4. FINANCIAL FEATURE SUMMARY")

financial_features = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
]

print(
    df[financial_features]
    .describe()
    .round(2)
    .to_string()
)

# --------------------------------------------------
# 6. Credit risk by income type
# --------------------------------------------------

print("\n5. REPAYMENT DIFFICULTY BY INCOME TYPE")

income_summary = (
    df.groupby("NAME_INCOME_TYPE", dropna=False)
    .agg(
        applications=("TARGET", "size"),
        repayment_difficulty_rate=("TARGET", "mean"),
    )
    .reset_index()
)

income_summary["repayment_difficulty_rate"] *= 100

print(
    income_summary.sort_values(
        "repayment_difficulty_rate",
        ascending=False,
    )
    .round(2)
    .to_string(index=False)
)

# --------------------------------------------------
# 7. Credit-to-income ratio
# --------------------------------------------------

print("\n6. CREDIT-TO-INCOME RATIO")

df["CREDIT_TO_INCOME"] = (
    df["AMT_CREDIT"]
    / df["AMT_INCOME_TOTAL"].where(df["AMT_INCOME_TOTAL"] > 0)
)

print(
    df.groupby("TARGET")["CREDIT_TO_INCOME"]
    .median()
    .round(2)
    .to_string()
)

# --------------------------------------------------
# 8. Save summary for future dashboard
# --------------------------------------------------

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

income_summary.to_csv(
    OUTPUT_DIR / "income_type_risk_summary.csv",
    index=False,
)

print("\nSaved: data/processed/income_type_risk_summary.csv")
print("\nData exploration completed successfully!")
