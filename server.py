"""JSON API + static host for the HTML/CSS/JS demo in web/.

Hypothesis: the browser cannot run the calibrated LightGBM pipeline or SHAP, so a
plain-JS frontend needs a thin Python layer that does exactly what app.py's
Decide tab does - and nothing more. If the two ever disagree on a score, this
file is the one that drifted.

Endpoints:
    GET  /api/meta   - threshold, cost assumptions, education options, presets
    POST /api/score  - one applicant -> calibrated P(default), decision,
                       expected profit, break-even risk, denial reasons,
                       what-if curve over loan size
    GET  /           - web/index.html (and the rest of web/)

Why it is built this way:
    - Loads ONLY artifacts/calibrated_pipeline.pkl, same as app.py. Profit needs
      true probabilities.
    - Never loads the test set. Test numbers would come from metrics.json.
    - Unset model inputs take the training median / mode (from load_train), so
      the applicant looks like someone the model has seen.
    - Artifacts load once at startup. If they are missing the server still
      starts and /api/meta returns the error, so the page can say what to run.

What breaks if an assumption is wrong:
    - If calibrated_pipeline.pkl and metrics.json come from different runs, the
      threshold and the scores disagree. Re-run train -> evaluate together.
    - applicant_rows() mirrors app.py. Changing which fields the form sets means
      changing both files.

Demo mode:
    LOAN_DEMO=1 with the artifacts missing swaps in demo_mock.py (a hand-written
    formula, NOT the model) so the UI can be walked through. /api/meta reports
    "demo": true and the page shows a banner. Real artifacts always win.

Run:  uvicorn server:app --host 127.0.0.1 --port 8000      then open http://127.0.0.1:8000
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.data import artifact_path, load_config, load_train, read_metrics
from src.evaluate import breakeven_threshold
from src.fairness import adverse_action_reasons

EDUCATION = [
    "Secondary / secondary special", "Higher education",
    "Incomplete higher", "Lower secondary", "Academic degree",
]
WHATIF_LOANS = np.geomspace(45_000, 4_000_000, 40)

cfg = load_config()
cost = cfg["cost_assumptions"]
lgd, interest = cost["loss_given_default"], cost["expected_interest"]
fixed = cost["fixed_cost_per_loan"] or 0


def load_state():
    """Everything the API needs, or an error string telling the user what to run."""
    metrics = read_metrics(cfg)
    pkl = Path(artifact_path(cfg, "calibrated_pipeline.pkl"))
    if "threshold" not in metrics or not pkl.exists():
        return None, ("Missing threshold or calibrated_pipeline.pkl in artifacts/. Copy artifacts/ from "
                      "the machine that trained the model, or run python -m src.train then python -m src.evaluate.")
    model = joblib.load(pkl)
    prep = model.calibrated_classifiers_[0].estimator["features"]["prep"]
    cats = set(next(cols for name, _, cols in prep.transformers_ if name == "cat"))
    X, _ = load_train(cfg)
    typical = {
        c: (X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else X[c].mode().iloc[0])
        for c in X.columns
    }
    return {"model": model, "cats": cats, "typical": typical,
            "threshold": metrics["threshold"]["value"]}, None


state, load_error = load_state()
demo = bool(load_error) and os.environ.get("LOAN_DEMO") == "1"
if demo:
    import demo_mock
    # Demo threshold = break-even at zero fixed cost, interest / (interest + lgd).
    state = {"threshold": interest / (interest + lgd), "typical": None}
    load_error = None


def num(v, default):
    return float(v) if pd.notna(v) else default


FIXED_PRESETS = {
    # Hypothesis: high external scores + longer employment land well below the threshold.
    "Strong file": {
        "income": 250_000, "credit": 450_000, "annuity": 22_000, "age": 42,
        "no_job": False, "employed": 12, "education": "Higher education",
        "ext1": 0.72, "ext2": 0.68, "ext3": 0.65,
    },
    # Hypothesis: low external scores + large loan vs income sit at or above the threshold.
    "High risk": {
        "income": 90_000, "credit": 900_000, "annuity": 48_000, "age": 24,
        "no_job": False, "employed": 1, "education": "Lower secondary",
        "ext1": 0.18, "ext2": 0.22, "ext3": 0.20,
    },
}


def typical_preset(typical):
    """The form values for a training-median applicant."""
    edu = typical.get("NAME_EDUCATION_TYPE", EDUCATION[0])
    days_emp = typical["DAYS_EMPLOYED"]
    return {
        "income": int(typical["AMT_INCOME_TOTAL"]),
        "credit": int(typical["AMT_CREDIT"]),
        "annuity": int(num(typical["AMT_ANNUITY"], 25_000)),
        "age": int(np.clip(-typical["DAYS_BIRTH"] / 365, 20, 70)),
        "no_job": bool(days_emp == 365243),
        "employed": 0 if days_emp == 365243 else int(np.clip(-days_emp / 365, 0, 45)),
        "education": edu if edu in EDUCATION else EDUCATION[0],
        "ext1": round(num(typical["EXT_SOURCE_1"], 0.5), 2),
        "ext2": round(num(typical["EXT_SOURCE_2"], 0.5), 2),
        "ext3": round(num(typical["EXT_SOURCE_3"], 0.5), 2),
    }


def presets():
    typical = dict(demo_mock.TYPICAL) if demo else typical_preset(state["typical"])
    return {"Typical": typical, **FIXED_PRESETS}


class Applicant(BaseModel):
    # Bounds match the Streamlit form so both demos accept the same inputs.
    income: float = Field(ge=10_000, le=10_000_000)
    credit: float = Field(ge=45_000, le=4_050_000)
    annuity: float = Field(ge=1_000, le=300_000)
    age: int = Field(ge=20, le=70)
    no_job: bool = False
    employed: int = Field(ge=0, le=45)
    education: str
    ext1: float = Field(ge=0, le=1)
    ext2: float = Field(ge=0, le=1)
    ext3: float = Field(ge=0, le=1)


def applicant_rows(a, loans):
    """One row per loan size; repayment scales with the loan. Other fields stay at training typical."""
    typical, cats, model = state["typical"], state["cats"], state["model"]
    rows = pd.DataFrame([typical] * len(loans))
    rows["AMT_INCOME_TOTAL"] = a.income
    rows["AMT_CREDIT"] = loans
    rows["AMT_GOODS_PRICE"] = loans
    rows["AMT_ANNUITY"] = a.annuity * np.asarray(loans) / a.credit
    rows["DAYS_BIRTH"] = -a.age * 365
    rows["DAYS_EMPLOYED"] = 365243 if a.no_job else -a.employed * 365
    rows["NAME_EDUCATION_TYPE"] = a.education
    for k, v in enumerate([a.ext1, a.ext2, a.ext3]):
        rows[f"EXT_SOURCE_{k + 1}"] = v
    cols = list(model.feature_names_in_)
    rows = rows[cols]
    return rows.astype({
        c: "float64" for c in cols
        if c not in cats and isinstance(typical[c], (int, float, np.number))
    })


app = FastAPI(title="Loan Risk Studio API")


@app.get("/api/meta")
def meta():
    if load_error:
        return {"ready": False, "error": load_error}
    return {
        "ready": True, "demo": demo, "threshold": state["threshold"],
        "lgd": lgd, "interest": interest, "fixed": fixed,
        "education": EDUCATION, "presets": presets(),
    }


def model_scores(a):
    """(P(default) for this loan, denial reasons if denied, P(default) over WHATIF_LOANS)."""
    threshold = state["threshold"]
    if demo:
        p = float(demo_mock.predict(a, [a.credit])[0])
        reasons = demo_mock.reasons(a) if p >= threshold else []
        return p, reasons, demo_mock.predict(a, WHATIF_LOANS)

    model = state["model"]
    row = applicant_rows(a, [a.credit])
    p = float(model.predict_proba(row)[0, 1])
    reasons = []
    if p >= threshold:
        for reason in adverse_action_reasons(model.calibrated_classifiers_[0].estimator, row)[0]:
            name, _, value = reason.partition(" = ")
            try:
                value = f"{float(value):,.4g}"
            except ValueError:
                pass
            reasons.append({"feature": name, "value": value})
    return p, reasons, model.predict_proba(applicant_rows(a, WHATIF_LOANS))[:, 1]


@app.post("/api/score")
def score(a: Applicant):
    if load_error:
        raise HTTPException(503, load_error)
    if a.education not in EDUCATION:
        raise HTTPException(422, f"education must be one of {EDUCATION}")
    threshold = state["threshold"]
    p, reasons, ps = model_scores(a)
    approved = p < threshold
    return {
        "demo": demo, "p": p, "approved": bool(approved), "threshold": threshold,
        "expected_profit": float(a.credit * ((1 - p) * interest - p * lgd) - fixed),
        "breakeven": float(max(breakeven_threshold(a.credit, lgd, interest, fixed), 0)),
        "reasons": reasons,
        "whatif": {
            "loans": WHATIF_LOANS.tolist(), "p": ps.tolist(),
            "profit": (WHATIF_LOANS * ((1 - ps) * interest - ps * lgd) - fixed).tolist(),
        },
    }


# Mounted last so /api/* routes win over static files.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True), name="web")
