
"""
CreditLens AI — Credit Risk Segmentation

Interactive portfolio risk analysis using
out-of-sample calibrated default probabilities.

Required data:
- Predicted probability of default
- Actual default outcome (0 or 1)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="CreditLens AI | Risk Segmentation",
    page_icon="📊",
    layout="wide",
)

st.title("Credit Portfolio Risk Segmentation")

st.caption(
    "CreditLens AI | Calibrated Default Probability Analysis"
)

st.write(
    "Explore borrower risk distributions, compare predicted "
    "and observed default rates, and assess portfolio "
    "risk concentration."
)


# --------------------------------------------------
# 1. LOAD MODEL PREDICTIONS
# --------------------------------------------------


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test_predictions.csv"
)

@st.cache_data
def load_predictions():
    return pd.read_csv(PREDICTIONS_PATH)

if not PREDICTIONS_PATH.exists():
    st.error(
        "Prediction file not found. "
        "Run the model calibration script first."
    )
    st.stop()

df = load_predictions()

st.success(
    f"Successfully loaded {len(df):,} test predictions."
)

if df.empty:
    st.error("The uploaded file contains no rows.")
    st.stop()

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

if not numeric_columns:
    st.error("No numeric columns were found.")
    st.stop()


def find_column(candidates, columns):
    for candidate in candidates:
        if candidate in columns:
            return columns.index(candidate)
    return 0


probability_index = find_column(
    [
        "calibrated_probability",
        "calibrated_pd",
        "predicted_pd",
        "pd",
        "y_prob",
        "y_pred_proba",
    ],
    numeric_columns,
)

actual_index = find_column(
    ["TARGET", "target", "y_true", "actual_default"],
    numeric_columns,
)

col1, col2 = st.columns(2)

with col1:
    probability_column = st.selectbox(
        "Calibrated Default Probability Column",
        numeric_columns,
        index=probability_index,
    )

with col2:
    actual_column = st.selectbox(
        "Actual Default Outcome Column",
        numeric_columns,
        index=actual_index,
    )

if probability_column == actual_column:
    st.error(
        "Select different columns for predicted "
        "probability and actual default outcome."
    )
    st.stop()

data = df[
    [probability_column, actual_column]
].copy()

data.columns = ["predicted_pd", "actual_default"]

data = data.replace(
    [np.inf, -np.inf], np.nan
).dropna()

if data.empty:
    st.error("No valid prediction rows remain.")
    st.stop()

if not data["predicted_pd"].between(0, 1).all():
    st.error(
        "Predicted probabilities must be between 0 and 1."
    )
    st.stop()

if not data["actual_default"].isin([0, 1]).all():
    st.error(
        "Actual default outcomes must contain only 0 or 1."
    )
    st.stop()


# --------------------------------------------------
# 2. RISK SEGMENTATION ASSUMPTIONS
# --------------------------------------------------

st.sidebar.header("Risk Segmentation Settings")

low_cutoff = st.sidebar.slider(
    "Low Risk Upper Bound (%)",
    min_value=1,
    max_value=15,
    value=5,
)

moderate_cutoff = st.sidebar.slider(
    "Moderate Risk Upper Bound (%)",
    min_value=5,
    max_value=25,
    value=10,
)

high_cutoff = st.sidebar.slider(
    "High Risk Upper Bound (%)",
    min_value=10,
    max_value=50,
    value=20,
)

if not low_cutoff < moderate_cutoff < high_cutoff:
    st.error(
        "Risk boundaries must be strictly increasing."
    )
    st.stop()

risk_order = [
    "Low Risk",
    "Moderate Risk",
    "High Risk",
    "Very High Risk",
]

data["risk_tier"] = pd.cut(
    data["predicted_pd"],
    bins=[
        -0.000001,
        low_cutoff / 100,
        moderate_cutoff / 100,
        high_cutoff / 100,
        1.0,
    ],
    labels=risk_order,
    include_lowest=True,
)

# --------------------------------------------------
# 3. PORTFOLIO SUMMARY
# --------------------------------------------------

total_applications = len(data)

average_pd = data["predicted_pd"].mean()

observed_default_rate = data["actual_default"].mean()

high_risk_count = data[
    data["risk_tier"].isin(
        ["High Risk", "Very High Risk"]
    )
].shape[0]

high_risk_share = (
    high_risk_count / total_applications
)

st.subheader("Portfolio Risk Overview")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Applications Analyzed",
    f"{total_applications:,}",
)

m2.metric(
    "Average Predicted PD",
    f"{average_pd:.2%}",
)

m3.metric(
    "Observed Default Rate",
    f"{observed_default_rate:.2%}",
)

m4.metric(
    "High / Very High Risk Share",
    f"{high_risk_share:.2%}",
)


# --------------------------------------------------
# 4. SEGMENTATION SUMMARY
# --------------------------------------------------

risk_summary = (
    data.groupby(
        "risk_tier",
        observed=False,
    )
    .agg(
        applications=("predicted_pd", "size"),
        average_predicted_pd=("predicted_pd", "mean"),
        observed_default_rate=(
            "actual_default", "mean"
        ),
        observed_defaults=("actual_default", "sum"),
    )
    .reindex(risk_order)
    .reset_index()
)

risk_summary["portfolio_share"] = (
    risk_summary["applications"] / total_applications
)

risk_summary["share_of_defaults"] = (
    risk_summary["observed_defaults"]
    / max(data["actual_default"].sum(), 1)
)

st.divider()

st.subheader("Portfolio Risk Distribution")

fig_distribution = px.bar(
    risk_summary,
    x="risk_tier",
    y="applications",
    color="risk_tier",
    category_orders={
        "risk_tier": risk_order
    },
    labels={
        "risk_tier": "Risk Segment",
        "applications": "Number of Applications",
    },
    title="Application Distribution by Risk Tier",
)

fig_distribution.update_layout(
    template="plotly_dark",
    showlegend=False,
    height=430,
)

st.plotly_chart(
    fig_distribution,
    use_container_width=True,
)


# --------------------------------------------------
# 5. PREDICTED VS OBSERVED DEFAULT RATE
# --------------------------------------------------

st.subheader("Predicted vs. Observed Default Risk")

comparison = risk_summary.melt(
    id_vars="risk_tier",
    value_vars=[
        "average_predicted_pd",
        "observed_default_rate",
    ],
    var_name="Metric",
    value_name="Rate",
)

comparison["Metric"] = comparison["Metric"].map({
    "average_predicted_pd": "Predicted PD",
    "observed_default_rate": "Observed Default Rate",
})

comparison["Rate (%)"] = comparison["Rate"] * 100

fig_comparison = px.bar(
    comparison,
    x="risk_tier",
    y="Rate (%)",
    color="Metric",
    barmode="group",
    category_orders={
        "risk_tier": risk_order
    },
    title="Predicted vs. Actual Default Rates",
    labels={"risk_tier": "Risk Segment"},
)

fig_comparison.update_layout(
    template="plotly_dark",
    height=430,
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True,
)

st.caption(
    "Differences between average predicted PD "
    "and observed default rates can indicate "
    "calibration differences within risk segments."
)


# --------------------------------------------------
# 6. PORTFOLIO RISK TABLE
# --------------------------------------------------

st.divider()

st.subheader("Risk Segmentation Summary")

st.dataframe(
    risk_summary.style.format({
        "applications": "{:,.0f}",
        "average_predicted_pd": "{:.2%}",
        "observed_default_rate": "{:.2%}",
        "observed_defaults": "{:,.0f}",
        "portfolio_share": "{:.2%}",
        "share_of_defaults": "{:.2%}",
    }, na_rep="N/A"),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download Risk Segmentation Summary",
    data=risk_summary.to_csv(index=False),
    file_name="credit_risk_segmentation.csv",
    mime="text/csv",
)


# --------------------------------------------------
# 7. AUTOMATED PORTFOLIO INSIGHTS
# --------------------------------------------------

st.divider()

st.subheader("Portfolio Risk Insights")

st.write(
    f"**{total_applications:,} applications** "
    f"were evaluated with an average predicted "
    f"default probability of **{average_pd:.2%}**."
)

st.write(
    f"**{high_risk_share:.2%}** of applications "
    "fall within the High or Very High Risk tiers."
)

calibration_gap = (
    observed_default_rate - average_pd
)

if abs(calibration_gap) < 0.01:
    st.success(
        "The portfolio-level predicted and observed "
        "default rates differ by less than "
        "1 percentage point."
    )
else:
    st.warning(
        "The portfolio-level predicted and observed "
        "default rates differ by "
        f"{abs(calibration_gap) * 100:.2f} "
        "percentage points."
    )

st.info(
    "Risk tiers are illustrative analytical groups, "
    "not automated lending approval decisions. "
    "Thresholds should be developed using validation "
    "data and business objectives, with final "
    "evaluation on a held-out test set."
)


# ==================================================
# 8. CREDIT RISK DECISION ANALYSIS
# ==================================================

st.divider()

st.header("Credit Risk Decision Analysis")

st.write(
    "Explore how different predicted default "
    "probability thresholds affect portfolio "
    "acceptance, default exposure, and risk capture."
)

st.caption(
    "Retrospective simulation using historical "
    "test-set outcomes. Not a production lending "
    "decision system."
)

# --------------------------------------------------
# A. USER-SELECTED RISK THRESHOLD
# --------------------------------------------------

st.subheader("Decision Threshold Simulator")

pd_threshold = st.slider(
    "Maximum Acceptable Predicted PD (%)",
    min_value=1,
    max_value=30,
    value=10,
    step=1,
    key="decision_pd_threshold",
)

threshold = pd_threshold / 100

# Hypothetical acceptance rule:
# Predicted PD <= threshold

accepted = data[
    data["predicted_pd"] <= threshold
]

flagged = data[
    data["predicted_pd"] > threshold
]

# --------------------------------------------------
# B. CALCULATE DECISION METRICS
# --------------------------------------------------

total_count = len(data)
accepted_count = len(accepted)
flagged_count = len(flagged)

total_defaults = int(
    data["actual_default"].sum()
)

flagged_defaults = int(
    flagged["actual_default"].sum()
)

acceptance_rate = (
    accepted_count / total_count
)

accepted_default_rate = (
    accepted["actual_default"].mean()
    if accepted_count > 0 else np.nan
)

default_capture_rate = (
    flagged_defaults / total_defaults
    if total_defaults > 0 else np.nan
)

average_accepted_pd = (
    accepted["predicted_pd"].mean()
    if accepted_count > 0 else np.nan
)

st.subheader("Decision Simulation Results")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Simulated Acceptance Rate",
    f"{acceptance_rate:.2%}",
)

m2.metric(
    "Accepted-Group Default Rate",
    (
        f"{accepted_default_rate:.2%}"
        if pd.notna(accepted_default_rate)
        else "N/A"
    ),
)

m3.metric(
    "Defaults Flagged",
    (
        f"{default_capture_rate:.2%}"
        if pd.notna(default_capture_rate)
        else "N/A"
    ),
)

m4.metric(
    "Average Accepted PD",
    (
        f"{average_accepted_pd:.2%}"
        if pd.notna(average_accepted_pd)
        else "N/A"
    ),
)

st.caption(
    f"Under this hypothetical rule, "
    f"{accepted_count:,} observations meet the "
    f"threshold and {flagged_count:,} are flagged."
)

# --------------------------------------------------
# C. THRESHOLD SENSITIVITY ANALYSIS
# --------------------------------------------------

st.divider()

st.subheader("Threshold Sensitivity Analysis")

threshold_results = []

for cutoff_pct in range(1, 31):

    cutoff = cutoff_pct / 100

    scenario_accepted = data[
        data["predicted_pd"] <= cutoff
    ]

    scenario_flagged = data[
        data["predicted_pd"] > cutoff
    ]

    n_accepted = len(scenario_accepted)

    scenario_defaults_flagged = (
        scenario_flagged["actual_default"].sum()
    )

    scenario_acceptance = (
        n_accepted / total_count
    )

    scenario_default_rate = (
        scenario_accepted["actual_default"].mean()
        if n_accepted > 0 else np.nan
    )

    scenario_capture = (
        scenario_defaults_flagged / total_defaults
        if total_defaults > 0 else np.nan
    )

    threshold_results.append({
        "Threshold (%)": cutoff_pct,
        "Acceptance Rate (%)":
            scenario_acceptance * 100,
        "Accepted Default Rate (%)":
            scenario_default_rate * 100,
        "Defaults Flagged (%)":
            scenario_capture * 100,
    })

threshold_df = pd.DataFrame(
    threshold_results
)

# --------------------------------------------------
# D. THRESHOLD TRADE-OFF CHART
# --------------------------------------------------

fig_threshold = go.Figure()

fig_threshold.add_trace(
    go.Scatter(
        x=threshold_df["Threshold (%)"],
        y=threshold_df["Acceptance Rate (%)"],
        mode="lines+markers",
        name="Acceptance Rate",
    )
)

fig_threshold.add_trace(
    go.Scatter(
        x=threshold_df["Threshold (%)"],
        y=threshold_df["Accepted Default Rate (%)"],
        mode="lines+markers",
        name="Accepted Default Rate",
    )
)

fig_threshold.add_trace(
    go.Scatter(
        x=threshold_df["Threshold (%)"],
        y=threshold_df["Defaults Flagged (%)"],
        mode="lines+markers",
        name="Defaults Flagged",
    )
)

fig_threshold.add_vline(
    x=pd_threshold,
    line_dash="dash",
    line_color="orange",
    annotation_text="Selected Threshold",
)

fig_threshold.update_layout(
    title="Risk Threshold Trade-Offs",
    xaxis_title="Maximum Predicted PD (%)",
    yaxis_title="Rate (%)",
    template="plotly_dark",
    height=450,
    hovermode="x unified",
)

st.plotly_chart(
    fig_threshold,
    use_container_width=True,
)

# --------------------------------------------------
# E. DECISION SUMMARY TABLE
# --------------------------------------------------

st.subheader("Threshold Comparison Table")

st.dataframe(
    threshold_df.style.format({
        "Threshold (%)": "{:.0f}%",
        "Acceptance Rate (%)": "{:.2f}%",
        "Accepted Default Rate (%)": "{:.2f}%",
        "Defaults Flagged (%)": "{:.2f}%",
    }, na_rep="N/A"),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download Threshold Analysis",
    data=threshold_df.to_csv(index=False),
    file_name="credit_decision_thresholds.csv",
    mime="text/csv",
)

# --------------------------------------------------
# F. AUTOMATED ANALYSIS
# --------------------------------------------------

st.divider()

st.subheader("Decision Analysis Insights")

st.write(
    f"At a maximum predicted PD of "
    f"**{pd_threshold}%**, the simulation "
    f"accepts **{acceptance_rate:.2%}** "
    "of test observations."
)

if pd.notna(accepted_default_rate):
    st.write(
        f"The accepted group has an observed "
        f"default rate of "
        f"**{accepted_default_rate:.2%}**."
    )

if pd.notna(default_capture_rate):
    st.write(
        f"The flagged group contains "
        f"**{default_capture_rate:.2%}** "
        "of the observed defaults."
    )

st.info(
    "Lower PD thresholds generally reduce "
    "simulated acceptance and credit risk "
    "exposure, while higher thresholds "
    "generally increase acceptance. "
    "The appropriate threshold depends on "
    "validated business objectives, costs, "
    "risk appetite, and applicable requirements."
)

st.caption(
    "This analysis uses held-out historical "
    "observations for illustration. Threshold "
    "selection should be performed on validation "
    "data, not optimized on final test results. "
    "Historical outcomes may also be subject to "
    "selection bias if only previously approved "
    "loans have observable defaults."
)
