
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ==================================================
# CREDITLENS AI
# Credit Default Prediction & Risk Analytics
# Modern Fintech Dashboard
# ==================================================


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="CreditLens AI | Credit Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_FILE = (
    BASE_DIR / "data" / "raw" / "application_train.csv"
)

PROCESSED_DIR = BASE_DIR / "data" / "processed"

INCOME_FILE = (
    PROCESSED_DIR / "income_type_risk_summary.csv"
)

THRESHOLD_FILE = (
    PROCESSED_DIR / "xgboost_threshold_analysis.csv"
)

MODEL_FILE = (
    BASE_DIR / "models" / "xgboost_model.joblib"
)


# --------------------------------------------------
# 2. Design system
# --------------------------------------------------

PRIMARY = "#4F46E5"
SECONDARY = "#8B83F4"
CORAL = "#F29A7B"
DARK = "#232641"
MUTED = "#777D92"
BACKGROUND = "#F3F5F9"
WHITE = "#FFFFFF"

st.markdown(
    """
    <style>

    /* Main application */

    .stApp {
        background-color: #F3F5F9;
        color: #232641;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E8EAF2;
    }

    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #4F46E5;
    }

    /* Headings */

    h1, h2, h3 {
        color: #232641;
        letter-spacing: -0.5px;
    }

    h1 {
        font-weight: 750;
    }

    /* Metrics */

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E8EAF2;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0px 3px 12px rgba(36, 39, 80, 0.03);
    }

    div[data-testid="stMetricLabel"] {
        color: #777D92;
    }

    div[data-testid="stMetricValue"] {
        color: #4F46E5;
        font-weight: 750;
    }

    /* Hero banner */

    .hero {
        background: linear-gradient(
            115deg,
            #4F46E5 0%,
            #796FF0 60%,
            #A099F8 100%
        );

        border-radius: 22px;
        padding: 35px 38px;
        margin-bottom: 30px;
        box-shadow: 0 10px 24px rgba(79, 70, 229, 0.12);
    }

    .hero h1 {
        color: white !important;
        font-size: 2.15rem;
        margin-bottom: 10px;
    }

    .hero p {
        color: #F0EEFF !important;
        font-size: 1rem;
        margin-bottom: 0;
        line-height: 1.7;
    }

    /* Small section heading */

    .section-label {
        color: #777D92;
        font-size: 12px;
        font-weight: 750;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    /* Custom result card */

    .result-card {
        background: #FFFFFF;
        border: 1px solid #E8EAF2;
        border-radius: 18px;
        padding: 22px 25px;
        margin: 12px 0;
    }

    .result-label {
        color: #777D92;
        font-size: 13px;
        font-weight: 600;
    }

    .result-value {
        color: #4F46E5;
        font-size: 44px;
        font-weight: 750;
        margin: 8px 0;
    }

    .result-description {
        color: #777D92;
        font-size: 13px;
        line-height: 1.6;
    }

    /* Buttons */

    .stButton > button {
        background-color: #4F46E5;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #4338CA;
        color: white;
    }

    /* Divider */

    hr {
        border-color: #E7EAF1;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. Data loading functions
# --------------------------------------------------

@st.cache_data
def load_portfolio_data():

    return pd.read_csv(
        RAW_FILE,
        usecols=["TARGET"],
    )


@st.cache_data
def load_income_data():

    return pd.read_csv(INCOME_FILE)


@st.cache_data
def load_threshold_data():

    return pd.read_csv(THRESHOLD_FILE)


@st.cache_resource
def load_xgboost_model(model_path, modified_time):

    return joblib.load(model_path)


# --------------------------------------------------
# 4. Check required data files
# --------------------------------------------------

required_data_files = [
    RAW_FILE,
    INCOME_FILE,
    THRESHOLD_FILE,
]

missing_files = [
    path
    for path in required_data_files
    if not path.exists()
]

if missing_files:

    st.error(
        "CreditLens AI cannot locate some required data files."
    )

    for file_path in missing_files:
        st.code(str(file_path))

    st.info(
        "Make sure you have completed the EDA and "
        "threshold analysis scripts."
    )

    st.stop()


# --------------------------------------------------
# 5. Load project data
# --------------------------------------------------

portfolio = load_portfolio_data()

income_data = load_income_data()

threshold_data = load_threshold_data()

total_applications = len(portfolio)

positive_cases = int(portfolio["TARGET"].sum())

negative_cases = (
    total_applications - positive_cases
)

observed_rate = portfolio["TARGET"].mean()


# --------------------------------------------------
# 6. Sidebar navigation
# --------------------------------------------------

with st.sidebar:

    st.markdown("## creditlens.")

    st.caption("INTELLIGENT CREDIT RISK ANALYTICS")

    st.divider()



    page = st.radio(
        "WORKSPACE",
        [
            "Portfolio Overview",
            "Model Comparison",
            "Threshold Explorer",
            "Risk Scenario Explorer",
            "Model Explainability",
            "Model Validation & Calibration",
        ],
    )


    st.divider()

    st.markdown("### About This Project")

    st.caption(
        "CreditLens AI combines Python, machine learning, "
        "and interactive financial analytics."
    )

    st.caption(
        "Dataset: Home Credit Default Risk"
    )

    st.caption(
        "Models: Logistic Regression & XGBoost"
    )

    st.divider()

    st.caption(
        "Educational research prototype only."
    )

    st.caption(
        "Not intended for real lending decisions."
    )


# --------------------------------------------------
# 7. Main header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>Understand risk. Make better decisions.</h1>
        <p>
            Explore historical credit outcomes,
            compare machine learning models,
            and investigate predictive risk signals.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# PAGE 1: PORTFOLIO OVERVIEW
# ==================================================

if page == "Portfolio Overview":

    st.markdown(
        '<p class="section-label">Portfolio Intelligence</p>',
        unsafe_allow_html=True,
    )

    st.title("Credit Portfolio Overview")

    st.write(
        "Explore historical credit application outcomes "
        "and repayment difficulty patterns."
    )

    st.write("")

    # ----------------------------------------------
    # KPI cards
    # ----------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Applications",
        f"{total_applications:,}",
    )

    col2.metric(
        "Observed Difficulty Rate",
        f"{observed_rate:.2%}",
    )

    col3.metric(
        "Difficulty Cases",
        f"{positive_cases:,}",
    )

    col4.metric(
        "No Difficulty",
        f"{negative_cases:,}",
    )

    st.write("")

    # ----------------------------------------------
    # Outcome distribution
    # ----------------------------------------------

    left, right = st.columns(2, gap="large")

    with left:

        st.subheader("Repayment Outcome Distribution")

        outcome_data = pd.DataFrame({
            "Outcome": [
                "No repayment difficulty",
                "Repayment difficulty",
            ],
            "Applications": [
                negative_cases,
                positive_cases,
            ],
        })

        fig = px.pie(
            outcome_data,
            names="Outcome",
            values="Applications",
            hole=0.65,
            color="Outcome",
            color_discrete_map={
                "No repayment difficulty": PRIMARY,
                "Repayment difficulty": CORAL,
            },
        )

        fig.update_traces(
            textinfo="percent",
            textposition="outside",
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color=DARK,
            margin=dict(
                l=15,
                r=15,
                t=20,
                b=15,
            ),
            legend_title_text="",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ----------------------------------------------
    # Income type analysis
    # ----------------------------------------------

    with right:

        st.subheader(
            "Repayment Difficulty by Income Type"
        )

        # Small categories are excluded from
        # the main chart to avoid misleading rates.

        chart_data = income_data[
            income_data["applications"] >= 100
        ].copy()

        chart_data = chart_data.sort_values(
            "repayment_difficulty_rate",
            ascending=True,
        )

        fig = px.bar(
            chart_data,
            x="repayment_difficulty_rate",
            y="NAME_INCOME_TYPE",
            orientation="h",
            color_discrete_sequence=[PRIMARY],
            labels={
                "repayment_difficulty_rate":
                    "Observed difficulty rate (%)",
                "NAME_INCOME_TYPE":
                    "Income type",
            },
            hover_data=["applications"],
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=DARK,
            margin=dict(
                l=15,
                r=15,
                t=20,
                b=15,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ----------------------------------------------
    # Income summary table
    # ----------------------------------------------

    st.divider()

    st.subheader("Historical Income Type Summary")

    display_income = income_data.copy()

    display_income = display_income.rename(
        columns={
            "NAME_INCOME_TYPE": "Income Type",
            "applications": "Applications",
            "repayment_difficulty_rate":
                "Repayment Difficulty Rate (%)",
        }
    )

    st.dataframe(
        display_income,
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "This dataset is imbalanced: approximately "
        "8% of applications have the positive target. "
        "Very small income categories should not be "
        "interpreted as reliable risk estimates."
    )


# ==================================================
# PAGE 2: MODEL COMPARISON
# ==================================================

elif page == "Model Comparison":

    st.markdown(
        '<p class="section-label">Machine Learning Lab</p>',
        unsafe_allow_html=True,
    )

    st.title("Model Performance Comparison")

    st.write(
        "Compare predictive performance between "
        "Logistic Regression and XGBoost."
    )

    # Results from previously executed scripts.
    # Replace these with generated metrics files
    # when the models are retrained.

    model_results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "XGBoost",
        ],
        "ROC-AUC": [
            0.6430,
            0.6656,
        ],
        "PR-AUC": [
            0.1375,
            0.1565,
        ],
    })

    st.write("")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Best ROC-AUC",
        "0.6656",
    )

    col2.metric(
        "Best PR-AUC",
        "0.1565",
    )

    col3.metric(
        "Best Ranking Model",
        "XGBoost",
    )

    st.write("")

    selected_metric = st.selectbox(
        "Performance Metric",
        [
            "ROC-AUC",
            "PR-AUC",
        ],
    )

    fig = px.bar(
        model_results,
        x="Model",
        y=selected_metric,
        color="Model",
        color_discrete_map={
            "Logistic Regression": SECONDARY,
            "XGBoost": PRIMARY,
        },
        text_auto=".4f",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=DARK,
        showlegend=False,
        yaxis_range=(
            [0, 1]
            if selected_metric == "ROC-AUC"
            else [0, 0.25]
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Model Performance Table")

    st.dataframe(
        model_results,
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    st.subheader("Model Interpretation")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Logistic Regression")

        st.write(
            "Provides an interpretable baseline "
            "with linear relationships between "
            "input features and model log-odds."
        )

        st.write(
            "**ROC-AUC:** 0.6430"
        )

    with col2:

        st.markdown("#### XGBoost")

        st.write(
            "Uses gradient-boosted decision trees "
            "to capture nonlinear relationships "
            "and interactions among features."
        )

        st.write(
            "**ROC-AUC:** 0.6656"
        )

    st.info(
        "These model metrics were obtained on the "
        "same historical test split. The models "
        "were developed using eight applicant features. "
        "This exploratory comparison is not a final "
        "independent validation."
    )


# ==================================================
# PAGE 3: THRESHOLD EXPLORER
# ==================================================

elif page == "Threshold Explorer":

    st.markdown(
        '<p class="section-label">Model Research</p>',
        unsafe_allow_html=True,
    )

    st.title("Classification Threshold Explorer")

    st.write(
        "Explore how changing the XGBoost classification "
        "threshold affects precision, recall, "
        "and flagged applications."
    )

    # ----------------------------------------------
    # Threshold selector
    # ----------------------------------------------

    available_thresholds = sorted(
        threshold_data["Threshold"]
        .unique()
        .tolist()
    )

    selected_threshold = st.select_slider(
        "Classification Threshold",
        options=available_thresholds,
        value=0.10 if 0.10 in available_thresholds
        else available_thresholds[0],
        format_func=lambda x: f"{x:.0%}",
    )

    selected_row = threshold_data.loc[
        np.isclose(
            threshold_data["Threshold"],
            selected_threshold,
        )
    ].iloc[0]

    st.write("")

    # ----------------------------------------------
    # Metrics
    # ----------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Precision",
        f"{selected_row['Precision']:.1%}",
    )

    col2.metric(
        "Recall",
        f"{selected_row['Recall']:.1%}",
    )

    col3.metric(
        "F1 Score",
        f"{selected_row['F1']:.3f}",
    )

    col4.metric(
        "Flagged Applications",
        f"{int(selected_row['Predicted Positive']):,}",
    )

    st.write("")

    # ----------------------------------------------
    # Performance chart
    # ----------------------------------------------

    st.subheader("Precision vs. Recall")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=threshold_data["Threshold"],
            y=threshold_data["Precision"],
            mode="lines+markers",
            name="Precision",
            line=dict(
                color=PRIMARY,
                width=3,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=threshold_data["Threshold"],
            y=threshold_data["Recall"],
            mode="lines+markers",
            name="Recall",
            line=dict(
                color=CORAL,
                width=3,
            ),
        )
    )

    fig.add_vline(
        x=selected_threshold,
        line_dash="dash",
        line_color=SECONDARY,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=DARK,
        xaxis_title="Classification Threshold",
        yaxis_title="Score",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        hovermode="x unified",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ----------------------------------------------
    # Classification results
    # ----------------------------------------------

    st.divider()

    st.subheader("Classification Outcomes")

    col1, col2 = st.columns(2)

    col1.metric(
        "True Positives",
        f"{int(selected_row['True Positive']):,}",
    )

    col2.metric(
        "False Positives",
        f"{int(selected_row['False Positive']):,}",
    )

    st.subheader("Threshold Comparison Table")

    st.dataframe(
        threshold_data,
        hide_index=True,
        use_container_width=True,
    )

    st.warning(
        "These results come from exploratory threshold "
        "analysis on the test set. Threshold selection "
        "must be performed using separate validation "
        "data before final model assessment."
    )


# ==================================================
# PAGE 4: RISK SCENARIO EXPLORER
# ==================================================

elif page == "Risk Scenario Explorer":

    st.markdown(
        '<p class="section-label">Interactive Risk Research</p>',
        unsafe_allow_html=True,
    )

    st.title("Risk Scenario Explorer")

    st.write(
        "Modify the financial characteristics of a "
        "fictional applicant to explore the behavior "
        "of the trained XGBoost model."
    )

    st.info(
        "This feature is for educational research only. "
        "Model outputs are not calibrated probabilities "
        "and should not be used for lending decisions."
    )

    # ----------------------------------------------
    # Check model
    # ----------------------------------------------

    if not MODEL_FILE.exists():

        st.error(
            "The XGBoost model file was not found."
        )

        st.code(str(MODEL_FILE))

        st.info(
            "Run python src/03_train_xgboost.py "
            "to create the trained model."
        )

        st.stop()

    model = load_xgboost_model(
        str(MODEL_FILE),
        MODEL_FILE.stat().st_mtime,
    )

    # ----------------------------------------------
    # Input and output columns
    # ----------------------------------------------

    left, right = st.columns(
        [1, 1],
        gap="large",
    )

    with left:

        st.subheader("Applicant Financial Profile")

        st.caption(
            "All characteristics are fictional. "
            "Monetary inputs use the original "
            "dataset's units, not USD."
        )

        income = st.number_input(
            "Total Income",
            min_value=25000.0,
            max_value=1000000.0,
            value=180000.0,
            step=10000.0,
        )

        credit = st.number_input(
            "Credit Amount",
            min_value=45000.0,
            max_value=2500000.0,
            value=500000.0,
            step=25000.0,
        )

        annuity = st.number_input(
            "Annuity Amount",
            min_value=1000.0,
            max_value=200000.0,
            value=25000.0,
            step=1000.0,
        )

        goods_price = st.number_input(
            "Goods Price",
            min_value=40000.0,
            max_value=2500000.0,
            value=450000.0,
            step=25000.0,
        )

        employment_years = st.slider(
            "Employment Duration (Years)",
            min_value=0,
            max_value=40,
            value=5,
        )

        income_type = st.selectbox(
            "Income Type",
            [
                "Working",
                "Commercial associate",
                "State servant",
                "Pensioner",
                "Businessman",
                "Student",
                "Unemployed",
                "Maternity leave",
            ],
        )

        education_type = st.selectbox(
            "Education",
            [
                "Secondary / secondary special",
                "Higher education",
                "Incomplete higher",
                "Lower secondary",
                "Academic degree",
            ],
        )

        contract_type = st.selectbox(
            "Loan Contract Type",
            [
                "Cash loans",
                "Revolving loans",
            ],
        )

    # ----------------------------------------------
    # Prepare applicant input
    # ----------------------------------------------

    applicant = pd.DataFrame(
        [
            {
                "AMT_INCOME_TOTAL": income,
                "AMT_CREDIT": credit,
                "AMT_ANNUITY": annuity,
                "AMT_GOODS_PRICE": goods_price,
                "DAYS_EMPLOYED":
                    -employment_years * 365.25,
                "NAME_INCOME_TYPE": income_type,
                "NAME_EDUCATION_TYPE": education_type,
                "NAME_CONTRACT_TYPE": contract_type,
            }
        ]
    )

    # ----------------------------------------------
    # Generate XGBoost model output
    # ----------------------------------------------

    risk_score = float(
        model.predict_proba(applicant)[0, 1]
    )

    credit_to_income = (
        credit / income
        if income > 0
        else np.nan
    )

    annuity_to_income = (
        annuity / income
        if income > 0
        else np.nan
    )

    # ----------------------------------------------
    # Display model results
    # ----------------------------------------------

    with right:

        st.subheader("Model Research Output")


        st.markdown(
            f"""
<div class="result-card">
    <div class="result-label">
        XGBoost Classification Score
    </div>
    <div class="result-value">
        {risk_score:.2%}
    </div>
    <div class="result-description">
        This score is generated by the saved XGBoost model.
        It is not a calibrated or validated individual
        default probability.
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )


        st.progress(
            min(max(risk_score, 0.0), 1.0)
        )

        st.caption(
            "The progress bar visualizes the model's "
            "numerical output on a 0–1 scale."
        )

        st.divider()

        st.subheader("Financial Ratios")

        ratio_col1, ratio_col2 = st.columns(2)

        ratio_col1.metric(
            "Credit-to-Income",
            f"{credit_to_income:.2f}x",
        )

        ratio_col2.metric(
            "Annuity-to-Income",
            f"{annuity_to_income:.2%}",
        )

        st.caption(
            "These ratios are contextual summaries. "
            "They are not additional features in "
            "the currently saved model."
        )

        st.divider()

        st.subheader("Scenario Input Summary")

        scenario_summary = pd.DataFrame({
            "Characteristic": [
                "Total Income",
                "Credit Amount",
                "Annuity",
                "Goods Price",
                "Employment Duration",
                "Income Type",
                "Education",
                "Contract Type",
            ],
            "Value": [
                f"{income:,.0f}",
                f"{credit:,.0f}",
                f"{annuity:,.0f}",
                f"{goods_price:,.0f}",
                f"{employment_years} years",
                income_type,
                education_type,
                contract_type,
            ],
        })

        st.dataframe(
            scenario_summary,
            hide_index=True,
            use_container_width=True,
        )

    # ----------------------------------------------
    # Model limitations
    # ----------------------------------------------

    st.divider()

    with st.expander(
        "How does this model work?",
        expanded=False,
    ):

        st.markdown(
            """
            ### Model Architecture

            The Risk Scenario Explorer uses a trained
            XGBoost classification model.

            The pipeline includes:

            - Missing-value imputation
            - One-hot encoding of categorical inputs
            - Gradient-boosted decision trees
            - Binary classification scoring

            ### Input Features

            The model uses eight applicant features:

            1. Total income
            2. Credit amount
            3. Annuity amount
            4. Goods price
            5. Employment duration
            6. Income type
            7. Education level
            8. Loan contract type

            ### Interpretation

            Changing an input recalculates the
            XGBoost model output.

            A higher score generally means the
            model ranks the scenario more strongly
            toward the positive class.

            Scores have not been probability-calibrated.

            ### Limitations

            - The data is historical.
            - The current model uses only eight features.
            - Scores are not validated for individual use.
            - Hypothetical scenarios can be unrealistic.
            - The tool does not recommend credit
              approval, denial, or loan pricing.
            """
        )


# ==================================================
# PAGE 5: MODEL EXPLAINABILITY
# ==================================================

elif page == "Model Explainability":

    st.markdown(
        '<p class="section-label">Explainable Machine Learning</p>',
        unsafe_allow_html=True,
    )

    st.title("Model Explainability")

    st.write(
        "Understand which financial and applicant "
        "characteristics contribute most to the "
        "trained XGBoost model's predictions."
    )

    # ----------------------------------------------
    # Load feature importance results
    # ----------------------------------------------

    importance_path = (
        PROCESSED_DIR / "xgboost_feature_importance.csv"
    )

    if not importance_path.exists():
        st.error(
            "Feature importance results were not found."
        )
        st.code("python src/05_feature_importance.py")
        st.stop()

    importance_df = pd.read_csv(importance_path)

    importance_df = importance_df.sort_values(
        "Relative_Importance",
        ascending=False,
    )

    # ----------------------------------------------
    # Overview metrics
    # ----------------------------------------------

    st.divider()

    top_feature = importance_df.iloc[0]

    top_three_importance = (
        importance_df["Relative_Importance"]
        .head(3)
        .sum()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Features Analyzed",
        f"{len(importance_df)}",
    )

    col2.metric(
        "Most Important Feature",
        top_feature["Feature_Label"],
    )

    col3.metric(
        "Top 3 Combined Importance",
        f"{top_three_importance:.1%}",
    )

    st.write("")

    # ----------------------------------------------
    # Feature importance chart
    # ----------------------------------------------

    st.subheader("Global Feature Importance")

    st.caption(
        "Relative importance based on the trained "
        "XGBoost model's built-in importance scores."
    )

    chart_df = importance_df.sort_values(
        "Relative_Importance",
        ascending=True,
    )

    fig = px.bar(
        chart_df,
        x="Relative_Importance",
        y="Feature_Label",
        orientation="h",
        color="Relative_Importance",
        color_continuous_scale=[
            "#DDD9FF",
            "#9F96F5",
            "#4F46E5",
        ],
        text="Relative_Importance",
        labels={
            "Relative_Importance": "Relative Importance",
            "Feature_Label": "Feature",
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_layout(
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=DARK,
        coloraxis_showscale=False,
        xaxis_tickformat=".0%",
        xaxis_range=[
            0,
            max(
                chart_df["Relative_Importance"].max()
                * 1.20,
                0.01,
            ),
        ],
        margin=dict(l=15, r=40, t=20, b=15),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ----------------------------------------------
    # Detailed feature ranking
    # ----------------------------------------------

    st.divider()

    st.subheader("Detailed Feature Rankings")

    display_df = importance_df[
        ["Feature_Label", "Relative_Importance"]
    ].copy()

    display_df["Relative_Importance"] = (
        display_df["Relative_Importance"] * 100
    )

    display_df.columns = [
        "Feature",
        "Relative Importance (%)",
    ]

    st.dataframe(
        display_df.round(2),
        hide_index=True,
        use_container_width=True,
    )

    # ----------------------------------------------
    # Model insights
    # ----------------------------------------------

    st.divider()

    st.subheader("What Did the Model Learn?")

    st.write(
        "The feature importance chart identifies "
        "which inputs the model uses most heavily "
        "according to its tree-based importance measure."
    )

    st.info(
        "Feature importance does not indicate "
        "causation or explain the direction of "
        "an individual prediction."
    )

    with st.expander("Feature Importance Methodology"):

        st.markdown(
            """
            **Model:** XGBoost Classifier

            **Importance method:** Built-in tree
            feature importance (gain-based).

            **Feature processing:**
            - Numeric values are imputed.
            - Categorical variables are one-hot encoded.
            - Importance values are aggregated by
              original applicant characteristic.
            - Grouped values are normalized to 100%.

            **Limitations:**
            - Importance is not a causal measure.
            - Correlated features can share importance.
            - One-hot aggregation can favor
              categorical variables with more levels.
            - Global importance does not explain
              individual applicant scores.
            - Model outputs are not validated
              for real lending decisions.

            A future improvement is to compare these
            rankings with permutation importance and
            SHAP explanations.
            """
        )


# ==================================================
# PAGE 6: MODEL VALIDATION & CALIBRATION
# ==================================================

elif page == "Model Validation & Calibration":

    st.markdown(
        '<p class="section-label">Model Reliability Lab</p>',
        unsafe_allow_html=True,
    )

    st.title("Model Validation & Calibration")

    st.write(
        "Evaluate probability reliability and compare "
        "raw XGBoost scores with isotonic calibration."
    )

    # ----------------------------------------------
    # Load validation results
    # ----------------------------------------------

    metrics_path = (
        PROCESSED_DIR / "calibration_metrics.csv"
    )

    curve_path = (
        PROCESSED_DIR / "calibration_curve.csv"
    )

    if not metrics_path.exists() or not curve_path.exists():
        st.error("Calibration results were not found.")
        st.code("python src/06_validate_and_calibrate.py")
        st.stop()

    metrics_df = pd.read_csv(metrics_path)
    curve_df = pd.read_csv(curve_path)

    raw = metrics_df.loc[
        metrics_df["Model"] == "XGBoost - Raw"
    ].iloc[0]

    calibrated = metrics_df.loc[
        metrics_df["Model"] == "XGBoost - Calibrated"
    ].iloc[0]

    # ----------------------------------------------
    # Main metrics
    # ----------------------------------------------

    st.subheader("Model Reliability Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Raw ROC-AUC",
        f"{raw['ROC_AUC']:.4f}",
    )

    col2.metric(
        "Raw PR-AUC",
        f"{raw['PR_AUC']:.4f}",
    )

    col3.metric(
        "Raw Brier Score",
        f"{raw['Brier_Score']:.5f}",
    )

    col4.metric(
        "Raw Log Loss",
        f"{raw['Log_Loss']:.5f}",
    )

    st.write("")

    # ----------------------------------------------
    # Calibration reliability curve
    # ----------------------------------------------

    st.divider()

    st.subheader("Predicted vs. Observed Outcomes")

    st.caption(
        "Each point represents a group of historical "
        "test observations. Points closer to the "
        "diagonal indicate better group-level calibration."
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0, 0.25],
            y=[0, 0.25],
            mode="lines",
            name="Perfect Calibration",
            line=dict(
                color="#A5A9BA",
                dash="dash",
                width=2,
            ),
        )
    )

    raw_curve = curve_df.loc[
        curve_df["Model"] == "XGBoost - Raw"
    ].sort_values("Mean_Predicted")

    calibrated_curve = curve_df.loc[
        curve_df["Model"] == "XGBoost - Calibrated"
    ].sort_values("Mean_Predicted")

    for data, label, color in [
        (raw_curve, "Raw XGBoost", "#4F46E5"),
        (
            calibrated_curve,
            "Isotonic Calibrated",
            "#F29A7B",
        ),
    ]:

        fig.add_trace(
            go.Scatter(
                x=data["Mean_Predicted"],
                y=data["Observed_Rate"],
                mode="lines+markers",
                name=label,
                line=dict(
                    color=color,
                    width=3,
                ),
                marker=dict(size=8),
                customdata=data[["Count"]],
                hovertemplate=(
                    "Predicted: %{x:.2%}<br>"
                    "Observed: %{y:.2%}<br>"
                    "Applications: %{customdata[0]:,}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.update_layout(
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=DARK,
        xaxis_title="Mean Predicted Score",
        yaxis_title="Observed Repayment Difficulty Rate",
        xaxis=dict(
            tickformat=".0%",
            range=[0, 0.25],
        ),
        yaxis=dict(
            tickformat=".0%",
            range=[0, 0.25],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ----------------------------------------------
    # Model comparison
    # ----------------------------------------------

    st.divider()

    st.subheader("Before vs. After Calibration")

    display_metrics = metrics_df.copy()

    display_metrics = display_metrics.rename(
        columns={
            "Model": "Model Version",
            "ROC_AUC": "ROC-AUC",
            "PR_AUC": "PR-AUC",
            "Brier_Score": "Brier Score",
            "Log_Loss": "Log Loss",
        }
    )

    st.dataframe(
        display_metrics.round(5),
        hide_index=True,
        use_container_width=True,
    )

    metric_options = {
        "ROC-AUC": "ROC_AUC",
        "PR-AUC": "PR_AUC",
        "Brier Score": "Brier_Score",
        "Log Loss": "Log_Loss",
    }

    selected_metric = st.selectbox(
        "Compare calibration performance by",
        list(metric_options.keys()),
    )

    metric_column = metric_options[selected_metric]

    fig_bar = px.bar(
        metrics_df,
        x="Model",
        y=metric_column,
        color="Model",
        text_auto=".5f",
        color_discrete_map={
            "XGBoost - Raw": "#4F46E5",
            "XGBoost - Calibrated": "#F29A7B",
        },
    )

    fig_bar.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=DARK,
        xaxis_title="Model Version",
        yaxis_title=selected_metric,
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True,
    )

    # ----------------------------------------------
    # Research interpretation
    # ----------------------------------------------

    st.divider()

    st.subheader("Research Findings")

    st.markdown(
        """
        **Key finding**

        Isotonic calibration did not improve the
        evaluated XGBoost model.

        The raw model produced slightly better:

        - ROC-AUC
        - PR-AUC
        - Brier Score
        - Log Loss

        Consequently, the original XGBoost model
        remains the preferred research baseline
        for this experiment.

        **Why is this interesting?**

        Probability calibration is not guaranteed
        to improve an already reasonably calibrated
        model. The results demonstrate the importance
        of testing changes rather than assuming
        additional complexity improves performance.
        """
    )

    with st.expander("Validation Methodology"):

        st.markdown(
            """
            **Development split**

            - 64% training
            - 16% calibration validation
            - 20% evaluation

            **Calibration method**

            Isotonic Regression

            **Evaluation metrics**

            - ROC-AUC measures ranking discrimination.
            - PR-AUC summarizes precision-recall behavior.
            - Brier Score measures probability error.
            - Log Loss penalizes incorrect predictions,
              especially confident errors.

            **Limitations**

            - The historical dataset may not represent
              a future lending population.
            - The evaluation partition was previously
              used during model exploration.
            - Calibration curves show group-level
              reliability, not guaranteed reliability
              for individual predictions.
            - No operational threshold has been approved.
            - Results are for educational research only.
            """
        )

    st.info(
        "The original XGBoost model remains the "
        "current research baseline. The calibrated "
        "model is retained for comparison, not "
        "automatically substituted into the "
        "Risk Scenario Explorer."
    )

# ==================================================
# APPLICATION FOOTER
# ==================================================

st.divider()

st.caption(
    "CreditLens AI | Python • Machine Learning • "
    "XGBoost • Streamlit | "
    "Educational Credit Risk Research"
)

st.caption(
    "Historical data and model outputs are not "
    "financial advice or lending decisions."
)
