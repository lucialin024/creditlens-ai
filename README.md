# CreditLens AI
### Credit Default Prediction & Explainable Risk Analytics

An interactive credit risk research platform built with **Python, XGBoost, Scikit-learn, Plotly, and Streamlit** to analyze historical repayment difficulties, compare predictive models, explore classification thresholds, and evaluate model reliability.

**Tech Stack:** Python | Pandas | Scikit-learn | XGBoost | Streamlit | Plotly | Machine Learning

## Project Overview

CreditLens AI is an independently developed machine learning project using the historical Home Credit Default Risk dataset.

The platform explores how predictive models can identify patterns associated with repayment difficulties while addressing important modeling challenges such as class imbalance, feature explainability, classification thresholds, and probability calibration.

The project combines predictive modeling with a modern fintech-style dashboard designed for interactive research and educational demonstrations.

## Dashboard Features

**1. Portfolio Overview**

Explore historical credit applications, observed repayment difficulty rates, and applicant characteristics.

**2. Model Comparison**

Compare Logistic Regression and XGBoost using ROC-AUC and average precision (PR-AUC).

**3. Classification Threshold Explorer**

Investigate how different classification thresholds affect precision, recall, F1 score, and false-positive counts.

**4. Risk Scenario Explorer**

Adjust fictional applicant characteristics and examine how a trained XGBoost model's score responds.

**5. Model Explainability**

Visualize XGBoost feature importance and explore the relative contributions of applicant characteristics.

**6. Model Validation & Calibration**

Compare raw XGBoost scores with isotonic-calibrated scores using reliability curves, ROC-AUC, PR-AUC, Brier Score, and Log Loss.

## Dataset

**Source:** Home Credit Default Risk — Kaggle

- 307,511 historical training applications
- 122 original dataset columns
- 24,825 positive repayment-difficulty cases
- 8.07% positive-class prevalence

The project uses a selected subset of eight applicant features for its initial predictive models.

## Machine Learning Results

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.6430 | 0.1375 |
| XGBoost | 0.6656 | 0.1565 |

XGBoost achieved better ranking discrimination than the baseline Logistic Regression model.

### Probability Calibration Experiment

A separate experiment evaluated isotonic calibration.

| Metric | Raw XGBoost | Isotonic Calibrated |
|---|---:|---:|
| ROC-AUC | 0.66587 | 0.66481 |
| PR-AUC | 0.15580 | 0.15146 |
| Brier Score | 0.07191 | 0.07199 |
| Log Loss | 0.26700 | 0.26817 |

Isotonic calibration did not improve performance on this evaluation partition. The raw model remained the preferred research baseline.

The calibration experiment used a 64% training, 16% calibration-validation, and 20% evaluation split. The evaluation partition had been explored in earlier development experiments and should not be considered a completely untouched final holdout.

## Feature Importance

The eight-feature XGBoost model's grouped, normalized built-in feature importance results were:

| Feature | Relative Importance |
|---|---:|
| Education Level | 37.10% |
| Income Type | 26.99% |
| Loan Contract Type | 12.87% |
| Employment Duration | 8.29% |
| Goods Price | 5.16% |
| Credit Amount | 4.43% |
| Annuity Amount | 3.70% |
| Total Income | 1.45% |

These values represent model importance weights, not causal effects or percentages of defaults explained.

## Project Structure

```text
creditlens-ai/
├── dashboard/
│   └── app.py
├── src/
│   ├── 01_explore_data.py
│   ├── 02_train_baseline.py
│   ├── 03_train_xgboost.py
│   ├── 04_evaluate_thresholds.py
│   ├── 05_feature_importance.py
│   └── 06_validate_and_calibrate.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── assets/
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── README.md
```

## Run Locally

Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/lucialin024/creditlens-ai.git
cd creditlens-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download `application_train.csv` from the Kaggle Home Credit Default Risk competition, subject to its access rules, and place it in `data/raw/`.

Run the project scripts in order:

```bash
python src/01_explore_data.py
python src/02_train_baseline.py
python src/03_train_xgboost.py
python src/04_evaluate_thresholds.py
python src/05_feature_importance.py
python src/06_validate_and_calibrate.py
```

Launch the dashboard:

```bash
python -m streamlit run dashboard/app.py
```

The local application requires its generated model and processed CSV artifacts. These are not included in the initial source-code repository.

**macOS note:** XGBoost may require the OpenMP runtime, installable through Homebrew with `brew install libomp`.

## Limitations and Future Improvements

- Use a genuinely untouched final evaluation dataset
- Evaluate additional features and model architectures
- Assess model performance and fairness across relevant populations
- Compare gain-based and permutation feature importance
- Introduce SHAP-based explanations
- Explore probability calibration and temporal validation
- Build a deployment-safe public demonstration

## Disclaimer

CreditLens AI is an educational machine learning and analytics prototype.

It is not designed or validated for actual lending approvals, denials, loan pricing, or individual creditworthiness decisions. Historical outcomes may not generalize to future applicants.

Dataset access, redistribution, and use remain subject to the original provider's terms.

## Author

**Lucia Lin**

Financial Analytics | Python | SQL | Machine Learning | Investment Research

[GitHub](https://github.com/lucialin024)