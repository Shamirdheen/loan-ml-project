"""Streamlit demo: score one applicant with the calibrated pipeline.

Loads ONLY artifacts/calibrated_pipeline.pkl (never the uncalibrated one) plus the
profit threshold from metrics.json. Fields the form does not ask for are left
missing; LightGBM handles missing values natively.

Run:  streamlit run app.py
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data import artifact_path, load_config, read_metrics
from src.fairness import adverse_action_reasons
from src.features import CATEGORICAL

cfg = load_config()
cost = cfg["cost_assumptions"]


@st.cache_resource
def load_model():
    return joblib.load(artifact_path(cfg, "calibrated_pipeline.pkl"))


def blank_row(columns):
    """One applicant with every training column present: categoricals None, numerics NaN."""
    return pd.DataFrame({c: [None] if c in CATEGORICAL else [np.nan] for c in columns}).astype(
        {c: "float64" for c in columns if c not in CATEGORICAL})


st.title("Loan approval risk")
metrics = read_metrics(cfg)
if "threshold" not in metrics:
    st.error("No threshold found. Run train.py and evaluate.py first.")
    st.stop()
threshold = metrics["threshold"]["value"]
model = load_model()

c1, c2 = st.columns(2)
income = c1.number_input("Annual income", 10_000, 5_000_000, 150_000, step=5_000)
credit = c1.number_input("Loan amount", 10_000, 4_000_000, 500_000, step=10_000)
annuity = c1.number_input("Yearly repayment (annuity)", 1_000, 300_000, 25_000, step=1_000)
age = c2.slider("Age (years)", 20, 70, 40)
employed = c2.slider("Years employed", 0, 40, 5)
ext = [c2.slider(f"External score {k}", 0.0, 1.0, 0.5) for k in (1, 2, 3)]
education = st.selectbox("Education", ["Secondary / secondary special", "Higher education",
                                       "Incomplete higher", "Lower secondary", "Academic degree"])

row = blank_row(model.feature_names_in_)
for col, val in {"AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
                 "DAYS_BIRTH": -age * 365, "DAYS_EMPLOYED": -employed * 365,
                 "EXT_SOURCE_1": ext[0], "EXT_SOURCE_2": ext[1], "EXT_SOURCE_3": ext[2],
                 "NAME_EDUCATION_TYPE": education}.items():
    if col in row.columns:
        row[col] = val

p = float(model.predict_proba(row)[0, 1])
expected = (1 - p) * cost["expected_interest"] * credit - p * cost["loss_given_default"] * credit
approved = p < threshold

st.metric("Probability of default (calibrated)", f"{p:.1%}")
st.write(f"Approval rule: approve if probability < {threshold:.1%}")
(st.success if approved else st.error)("APPROVE" if approved else "DENY")
st.write(f"Expected profit if approved: {expected:,.0f}")

if not approved:
    # One fitted fold model from the calibrated ensemble; calibration is monotone,
    # so it ranks the same risk drivers as the full model.
    fold_model = model.calibrated_classifiers_[0].estimator
    st.subheader("Main reasons for denial")
    for reason in adverse_action_reasons(fold_model, row)[0]:
        st.write("- " + reason)
