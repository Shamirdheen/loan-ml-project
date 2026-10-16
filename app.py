"""Streamlit faculty demo: score an applicant and walk the four grade-earning decisions.

Four tabs, in the order a live demo should go:
    Decide      - form -> calibrated P(default), APPROVE/DENY, expected profit,
                  denial reasons, what-if over loan size.
    Model       - test metrics from metrics.json (never recomputed), PR-AUC
                  ladder, reliability curve, printable depth-4 tree.
    Profit lab  - LGD / interest / fixed cost sliders on TRAINING out-of-fold
                  predictions. The deployed threshold does not move with sliders.
    Fairness    - equalized odds by CODE_GENDER on training OOF + the saved
                  CODE_GENDER ablation.

Why it is built this way:
    - Loads ONLY artifacts/calibrated_pipeline.pkl. Profit needs true
      probabilities; LightGBM's class-weighted scores are not those.
    - Never loads the test set. Test numbers are read from metrics.json.
      Interactive sliders use oof_predictions.csv (training split).
    - Unset form fields take the training median / mode. An applicant with
      ~110 missing fields is unlike anyone the model saw.
    - Categorical columns come from the fitted pipeline (dtype selector at
      fit time), so the form cannot drift from training.

What breaks if an assumption is wrong:
    - If metrics.json / .pkl / oof_predictions.csv are from different runs,
      the deployed threshold and the curves disagree. Re-run train ->
      evaluate -> fairness together.
    - The profit lab's optimum is an exploration aid. Changing the deployed
      threshold means changing config.yaml and re-running evaluate.py.

Run:  streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import artifact_path, load_config, load_train, read_metrics
from src.evaluate import breakeven_threshold
from src.fairness import MIN_GROUP, adverse_action_reasons, gaps, group_table

BLUE, ORANGE, MUTED, AXIS = "#1b4f8a", "#c45c26", "#7a756c", "#d4cfc4"
GOOD, CRITICAL = "#0f7a3a", "#c0392b"
GRID = np.linspace(0.01, 0.99, 197)
EDUCATION = [
    "Secondary / secondary special", "Higher education",
    "Incomplete higher", "Lower secondary", "Academic degree",
]

st.set_page_config(
    page_title="Loan Risk Studio",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)
cfg = load_config()
cost = cfg["cost_assumptions"]
id_col, prot = cfg["data"]["id_column"], cfg["data"]["protected_attribute"]


# ---------------------------------------------------------------- chrome
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="st-"] { font-family: "IBM Plex Sans", sans-serif; }
.block-container { padding-top: 1.15rem; padding-bottom: 2rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: #faf8f3; border-right: 1px solid #e6e0d4; }
div[data-testid="stMetric"] {
    background: #fff; border: 1px solid #e6e0d4; border-radius: 14px;
    padding: 12px 16px;
}
div[data-testid="stMetric"] label { color: #7a756c; letter-spacing: .02em; }
.hero {
    background: linear-gradient(135deg, #132a4a 0%, #1b4f8a 70%);
    color: #fff; border-radius: 18px; padding: 22px 26px; margin-bottom: 18px;
}
.hero h1 { color: #fff !important; font-size: 1.85rem; margin: 0 0 4px 0; }
.hero p { color: rgba(255,255,255,.82); margin: 0; font-size: .95rem; }
.verdict {
    border-radius: 16px; padding: 18px 20px; margin: 8px 0 16px;
    border: 1px solid transparent;
}
.verdict .kicker { font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; font-weight: 600; }
.verdict .headline { font-size: 2.1rem; font-weight: 700; line-height: 1.1; margin: 4px 0; }
.verdict .sub { opacity: .85; }
.verdict.ok { background: #e8f6ec; border-color: #b7e0c2; color: #0f7a3a; }
.verdict.no { background: #fdecea; border-color: #f0b8b2; color: #c0392b; }
.formula {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    background: #fff; border: 1px dashed #d4cfc4; border-radius: 10px;
    padding: 10px 14px; font-size: .86rem; color: #3f3a34;
}
.hint { color: #7a756c; font-size: .88rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading calibrated model…")
def load_model():
    return joblib.load(artifact_path(cfg, "calibrated_pipeline.pkl"))


@st.cache_resource(show_spinner="Loading training view (first run only)…")
def load_training_view():
    """Train-split OOF predictions joined to loan size and gender, plus a typical applicant."""
    X, _ = load_train(cfg)
    oof = pd.read_csv(artifact_path(cfg, "oof_predictions.csv"))
    oof = oof.merge(X[[id_col, cost["exposure_column"], prot]], on=id_col, how="left")
    typical = {
        c: (X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else X[c].mode().iloc[0])
        for c in X.columns
    }
    return oof, typical


def categorical_columns(model):
    """Categorical inputs the fitted pipeline actually used (picked by dtype at fit)."""
    prep = model.calibrated_classifiers_[0].estimator["features"]["prep"]
    return set(next(cols for name, _, cols in prep.transformers_ if name == "cat"))


def plot(fig, height=320):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=44, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color="#1c1917"),
        legend=dict(orientation="h", y=1.14, x=0), hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS)
    fig.update_yaxes(gridcolor="rgba(122,117,108,0.18)", zeroline=False)
    st.plotly_chart(fig, width="stretch", theme=None)


def profit_curve(y, p, exposure, lgd, interest, fixed):
    """Profit at every threshold in GRID in one pass (sort by p, cumulative sum)."""
    order = np.argsort(p)
    gain = (np.where(y == 0, interest * exposure, -lgd * exposure) - fixed)[order]
    csum = np.concatenate([[0.0], np.cumsum(gain)])
    n_approved = np.searchsorted(p[order], GRID, side="left")
    return csum[n_approved], n_approved / len(p)


def figure_or_note(name, caption):
    path = artifact_path(cfg, f"figures/{name}")
    if Path(path).exists():
        st.image(str(path), width="stretch", caption=caption)
    else:
        st.info(f"Missing `artifacts/figures/{name}`. Re-run the matching module to generate it.")


def apply_preset(name, typical):
    edu = typical.get("NAME_EDUCATION_TYPE", EDUCATION[0])
    if edu not in EDUCATION:
        edu = EDUCATION[0]
    presets = {
        "Typical": {
            "income": int(typical["AMT_INCOME_TOTAL"]),
            "credit": int(typical["AMT_CREDIT"]),
            "annuity": int(typical["AMT_ANNUITY"] if pd.notna(typical["AMT_ANNUITY"]) else 25_000),
            "age": int(np.clip(-typical["DAYS_BIRTH"] / 365, 20, 70)),
            "no_job": bool(typical["DAYS_EMPLOYED"] == 365243),
            "employed": 0 if typical["DAYS_EMPLOYED"] == 365243
            else int(np.clip(-typical["DAYS_EMPLOYED"] / 365, 0, 45)),
            "education": edu,
            "ext1": round(float(typical["EXT_SOURCE_1"]), 2) if pd.notna(typical["EXT_SOURCE_1"]) else 0.50,
            "ext2": round(float(typical["EXT_SOURCE_2"]), 2) if pd.notna(typical["EXT_SOURCE_2"]) else 0.50,
            "ext3": round(float(typical["EXT_SOURCE_3"]), 2) if pd.notna(typical["EXT_SOURCE_3"]) else 0.50,
        },
        # Hypothesis: high external scores + longer employment should land well below the 0.19 threshold.
        "Strong file": {
            "income": 250_000, "credit": 450_000, "annuity": 22_000, "age": 42,
            "no_job": False, "employed": 12, "education": "Higher education",
            "ext1": 0.72, "ext2": 0.68, "ext3": 0.65,
        },
        # Hypothesis: low external scores + large loan vs income should sit at or above the threshold.
        "High risk": {
            "income": 90_000, "credit": 900_000, "annuity": 48_000, "age": 24,
            "no_job": False, "employed": 1, "education": "Lower secondary",
            "ext1": 0.18, "ext2": 0.22, "ext3": 0.20,
        },
    }
    st.session_state.update(presets[name])


# ---------------------------------------------------------------- guards
metrics = read_metrics(cfg)
st.markdown(
    """<div class="hero">
    <h1>Loan Risk Studio</h1>
    <p>Home Credit default-risk model · calibrated LightGBM · expected-profit decision rule.
    Test-set numbers are read from disk — nothing interactive retouches the held-out set.</p>
    </div>""",
    unsafe_allow_html=True,
)
if "threshold" not in metrics or "test" not in metrics:
    st.error(
        "No threshold or test results found. Run `python -m src.train`, then "
        "`python -m src.evaluate`, then `python -m src.fairness` before the demo."
    )
    st.stop()

pkl = artifact_path(cfg, "calibrated_pipeline.pkl")
oof_path = artifact_path(cfg, "oof_predictions.csv")
if not Path(pkl).exists() or not Path(oof_path).exists():
    st.error("Missing `calibrated_pipeline.pkl` or `oof_predictions.csv` in artifacts/. Run evaluate.py first.")
    st.stop()

threshold = metrics["threshold"]["value"]
lgd, interest, fixed = cost["loss_given_default"], cost["expected_interest"], cost["fixed_cost_per_loan"] or 0
model = load_model()
cats = categorical_columns(model)
oof, typical = load_training_view()

if "income" not in st.session_state:
    apply_preset("Typical", typical)

h = st.columns(4)
h[0].metric("Approval threshold", f"{threshold:.1%}",
            help="Approve if calibrated P(default) is below this. Chosen on training out-of-fold predictions.")
h[1].metric("Loss given default", f"{lgd:.0%}")
h[2].metric("Interest if repaid", f"{interest:.0%}")
h[3].metric("Fixed cost / loan", f"{fixed:,.0f}")

tab_decide, tab_model, tab_profit, tab_fair = st.tabs(
    ["1 · Decide", "2 · Model", "3 · Profit lab", "4 · Fairness"]
)

# ================================================================= Decide
with tab_decide:
    form, result = st.columns([1, 1.2], gap="large")
    with form:
        st.subheader("Applicant")
        st.caption("Unset fields take the training median or mode. Gender is not a model input.")
        p1, p2, p3 = st.columns(3)
        if p1.button("Typical", use_container_width=True):
            apply_preset("Typical", typical)
            st.rerun()
        if p2.button("Strong file", use_container_width=True):
            apply_preset("Strong file", typical)
            st.rerun()
        if p3.button("High risk", use_container_width=True):
            apply_preset("High risk", typical)
            st.rerun()

        c1, c2 = st.columns(2)
        income = c1.number_input("Annual income", 10_000, 10_000_000, step=5_000, key="income")
        credit = c1.number_input("Loan amount", 45_000, 4_050_000, step=10_000, key="credit")
        annuity = c1.number_input("Yearly repayment (annuity)", 1_000, 300_000, step=1_000, key="annuity")
        age = c2.slider("Age (years)", 20, 70, key="age")
        no_job = c2.checkbox("Pensioner / not employed", key="no_job")
        employed = c2.slider("Years employed", 0, 45, disabled=no_job, key="employed")
        education = c2.selectbox("Education", EDUCATION, key="education")
        st.markdown("**External credit scores** (0 = worst, 1 = best)")
        e = st.columns(3)
        ext = [
            e[0].slider("Score 1", 0.0, 1.0, step=0.01, key="ext1"),
            e[1].slider("Score 2", 0.0, 1.0, step=0.01, key="ext2"),
            e[2].slider("Score 3", 0.0, 1.0, step=0.01, key="ext3"),
        ]

    def applicant_rows(loans):
        """One row per loan size; repayment scales with the loan. Other fields stay at training typical."""
        rows = pd.DataFrame([typical] * len(loans))
        rows["AMT_INCOME_TOTAL"] = income
        rows["AMT_CREDIT"] = loans
        rows["AMT_GOODS_PRICE"] = loans
        rows["AMT_ANNUITY"] = annuity * np.asarray(loans) / credit
        rows["DAYS_BIRTH"] = -age * 365
        rows["DAYS_EMPLOYED"] = 365243 if no_job else -employed * 365
        rows["NAME_EDUCATION_TYPE"] = education
        for k in range(3):
            rows[f"EXT_SOURCE_{k + 1}"] = ext[k]
        cols = list(model.feature_names_in_)
        rows = rows[cols]
        return rows.astype({
            c: "float64" for c in cols
            if c not in cats and isinstance(typical[c], (int, float, np.number))
        })

    row = applicant_rows([credit])
    p = float(model.predict_proba(row)[0, 1])
    approved = p < threshold
    expected = credit * ((1 - p) * interest - p * lgd) - fixed
    be = max(breakeven_threshold(credit, lgd, interest, fixed), 0)

    with result:
        st.subheader("Decision")
        kind, label, icon = ("ok", "APPROVE", "✓") if approved else ("no", "DENY", "✕")
        rel = "below" if approved else "at or above"
        st.markdown(
            f"""<div class="verdict {kind}">
            <div class="kicker">Live decision · calibrated LightGBM</div>
            <div class="headline">{icon} {label}</div>
            <div class="sub">Default risk {p:.1%} is {rel} the {threshold:.1%} profit threshold.</div>
            </div>""",
            unsafe_allow_html=True,
        )
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100,
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 36}},
            title={"text": "Calibrated P(default)"},
            gauge={
                "axis": {"range": [0, max(40, p * 100 + 8)], "ticksuffix": "%"},
                "bar": {"color": CRITICAL if not approved else GOOD, "thickness": 0.28},
                "steps": [
                    {"range": [0, threshold * 100], "color": "rgba(15,122,58,0.16)"},
                    {"range": [threshold * 100, 100], "color": "rgba(192,57,43,0.12)"},
                ],
                "threshold": {"line": {"color": MUTED, "width": 3}, "value": threshold * 100},
            },
        ))
        g.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=0),
                        paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(g, width="stretch", theme=None)

        m = st.columns(2)
        m[0].metric("Expected profit if approved", f"{expected:,.0f}",
                    delta="profitable" if expected > 0 else "loss-making",
                    delta_color="normal" if expected > 0 else "inverse")
        m[1].metric("Break-even risk for this loan", f"{be:.1%}",
                    help="Risk at which expected profit is exactly zero for this loan size.")
        st.markdown(
            f'<div class="formula">E[profit] = {credit:,.0f} × ((1 − p)×{interest:.2f} − p×{lgd:.2f})'
            f" − {fixed:,.0f} &nbsp;→&nbsp; approve if p &lt; {threshold:.2f}</div>",
            unsafe_allow_html=True,
        )
        if not approved:
            st.markdown("**Main reasons for denial** (features pushing risk up most)")
            fold_model = model.calibrated_classifiers_[0].estimator
            reasons = adverse_action_reasons(fold_model, row)[0]
            if reasons:
                for reason in reasons:
                    name, _, value = reason.partition(" = ")
                    try:
                        value = f"{float(value):,.4g}"
                    except ValueError:
                        pass
                    st.markdown(f"- `{name}` = {value}")
            else:
                st.caption("No single feature dominates this denial.")

    st.divider()
    st.subheader("What if the loan were a different size?")
    st.caption("Same applicant; repayment scales with the loan. Threshold stays the deployed one.")
    loans = np.geomspace(45_000, 4_000_000, 40)
    ps = model.predict_proba(applicant_rows(loans))[:, 1]
    w1, w2 = st.columns(2)
    with w1:
        f = go.Figure()
        f.add_scatter(x=loans, y=ps, name="this applicant", line=dict(color=BLUE, width=2.5),
                      hovertemplate="%{y:.1%}<extra></extra>")
        f.add_hline(y=threshold, line=dict(color=MUTED, dash="dash"), annotation_text="approval threshold")
        f.add_scatter(x=[credit], y=[p], mode="markers", name="current loan",
                      marker=dict(size=12, color=ORANGE, line=dict(width=2, color="white")),
                      hovertemplate="%{y:.1%}<extra></extra>")
        f.update_xaxes(type="log", title="loan amount")
        f.update_yaxes(tickformat=".0%", title="P(default)")
        f.update_layout(title="Default risk vs loan size")
        plot(f)
    with w2:
        prof = loans * ((1 - ps) * interest - ps * lgd) - fixed
        f = go.Figure()
        f.add_scatter(x=loans, y=prof, name="expected profit", line=dict(color=BLUE, width=2.5),
                      hovertemplate="%{y:,.0f}<extra></extra>")
        f.add_hline(y=0, line=dict(color=AXIS))
        f.add_scatter(x=[credit], y=[expected], mode="markers", name="current loan",
                      marker=dict(size=12, color=ORANGE, line=dict(width=2, color="white")),
                      hovertemplate="%{y:,.0f}<extra></extra>")
        f.update_xaxes(type="log", title="loan amount")
        f.update_yaxes(title="expected profit")
        f.update_layout(title="Expected profit vs loan size", showlegend=False)
        plot(f)

# ================================================================= Model
with tab_model:
    test = metrics["test"]
    st.subheader("Held-out test set (evaluated once)")
    st.caption("These numbers are read from metrics.json. This tab does not load or score the test set.")
    t = st.columns(4)
    t[0].metric("PR-AUC (primary)", f"{test['calibrated']['pr_auc']:.3f}",
                help=f"Base rate is {oof.y.mean():.1%}; a random model scores about that.")
    t[1].metric("ROC-AUC", f"{test['calibrated']['roc_auc']:.3f}")
    t[2].metric("Brier score", f"{test['calibrated']['brier_score']:.4f}",
                delta=f"{test['calibrated']['brier_score'] - test['uncalibrated']['brier_score']:+.4f} vs uncalibrated",
                delta_color="inverse")
    t[3].metric("Approval rate", f"{test['approval_rate']:.1%}")
    uplift = test["profit_at_threshold"] - test["profit_approve_all"]
    st.info(
        f"At the profit threshold the test set earns **{test['profit_at_threshold']:,.0f}**, "
        f"versus **{test['profit_approve_all']:,.0f}** for approving everyone "
        f"(**{uplift:+,.0f}**, {uplift / abs(test['profit_approve_all']):+.1%})."
    )

    l, r = st.columns(2)
    with l:
        cv = pd.DataFrame(metrics["cv"]).T.reset_index(names="model").sort_values("pr_auc")
        f = go.Figure(go.Bar(
            x=cv.pr_auc, y=cv.model, orientation="h",
            marker_color=[BLUE if mdl == "lightgbm" else AXIS for mdl in cv.model],
            text=[f"{v:.3f}" for v in cv.pr_auc], textposition="outside",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        f.add_vline(x=oof.y.mean(), line=dict(color=MUTED, dash="dot"), annotation_text="base rate")
        f.update_layout(title="5-model ladder · 5-fold CV PR-AUC (train split)", hovermode="closest")
        f.update_xaxes(range=[0, cv.pr_auc.max() * 1.22], title="PR-AUC")
        plot(f, 340)
        if "accuracy" in metrics["cv"].get("baseline", {}):
            st.caption(
                f"Why not accuracy: predicting nobody defaults already scores "
                f"{metrics['cv']['baseline']['accuracy']:.1%} accuracy and zero useful predictions."
            )
    with r:
        st.markdown("**Calibration: before vs after isotonic**")
        figure_or_note("reliability.png", "Uncalibrated LightGBM is over-confident because of class weighting.")
    with st.expander("Printable depth-4 decision tree"):
        figure_or_note("decision_tree.png", "The tree is a baseline for the report figure, not the deployed model.")

# ================================================================= Profit lab
with tab_profit:
    st.subheader("Profit lab")
    st.caption(
        "Explore cost assumptions on **training out-of-fold** predictions. The test set is not used. "
        "The deployed threshold only changes when config.yaml changes and evaluate.py is re-run."
    )
    s = st.columns(3)
    lab_lgd = s[0].slider("Loss given default", 0.1, 1.0, float(lgd), 0.05)
    lab_int = s[1].slider("Interest earned if repaid", 0.02, 0.40, float(interest), 0.01)
    lab_fix = s[2].number_input("Fixed cost per loan", 0, 200_000, int(fixed), step=1_000)

    y_o, p_o, ex_o = oof.y.to_numpy(), oof.p.to_numpy(), oof[cost["exposure_column"]].to_numpy()
    curve, approval = profit_curve(y_o, p_o, ex_o, lab_lgd, lab_int, lab_fix)
    best = int(curve.argmax())
    deployed = int(np.abs(GRID - threshold).argmin())
    k = st.columns(4)
    k[0].metric("Best threshold for these costs", f"{GRID[best]:.1%}",
                delta=f"{GRID[best] - threshold:+.1%} vs deployed", delta_color="off")
    k[1].metric("Profit at best", f"{curve[best]:,.0f}")
    k[2].metric("Profit at deployed threshold", f"{curve[deployed]:,.0f}")
    k[3].metric("Approval rate at best", f"{approval[best]:.1%}")

    a, b = st.columns(2)
    with a:
        f = go.Figure()
        f.add_scatter(x=GRID, y=curve, name="profit", line=dict(color=BLUE, width=2.5),
                      hovertemplate="%{y:,.0f}<extra></extra>")
        f.add_vline(x=GRID[best], line=dict(color=ORANGE, dash="dash"), annotation_text="best")
        f.add_vline(x=threshold, line=dict(color=MUTED, dash="dot"), annotation_text="deployed",
                    annotation_position="bottom right")
        f.update_xaxes(title="threshold (approve if P(default) < t)", tickformat=".0%")
        f.update_yaxes(title="total profit (training OOF)")
        f.update_layout(title="Profit vs threshold", showlegend=False)
        plot(f)
    with b:
        f = go.Figure()
        f.add_scatter(x=GRID, y=approval, name="approval rate", line=dict(color=BLUE, width=2.5),
                      hovertemplate="%{y:.1%}<extra></extra>")
        f.add_vline(x=GRID[best], line=dict(color=ORANGE, dash="dash"))
        f.update_xaxes(title="threshold", tickformat=".0%")
        f.update_yaxes(title="share approved", tickformat=".0%", range=[0, 1.02])
        f.update_layout(title="Approval rate vs threshold", showlegend=False)
        plot(f)

    st.markdown("**Break-even threshold by loan size** — with a fixed cost, small loans need lower risk to pay off")
    sizes = np.geomspace(max(lab_fix / lab_int * 0.5, 20_000) if lab_fix else 20_000, 4_000_000, 200)
    f = go.Figure()
    f.add_scatter(
        x=sizes, y=np.clip(breakeven_threshold(sizes, lab_lgd, lab_int, lab_fix), 0, None),
        name="break-even threshold", line=dict(color=BLUE, width=2.5), hovertemplate="%{y:.1%}<extra></extra>",
    )
    f.add_hline(y=GRID[best], line=dict(color=MUTED, dash="dash"), annotation_text="best single threshold")
    lo, hi = np.percentile(ex_o, [5, 95])
    f.add_vrect(x0=lo, x1=hi, fillcolor=MUTED, opacity=0.12, line_width=0,
                annotation_text="middle 90% of loans", annotation_position="top left")
    f.update_xaxes(type="log", title="loan amount")
    f.update_yaxes(tickformat=".0%", title="approve if P(default) <", rangemode="tozero")
    f.update_layout(showlegend=False)
    plot(f, 300)

# ================================================================= Fairness
with tab_fair:
    st.subheader("Fairness by CODE_GENDER")
    st.caption(
        f"Primary metric: **equalized odds** — repayers should be approved at the same rate in every "
        f"group, and so should defaulters. Training out-of-fold only. Groups under {MIN_GROUP} people "
        "are listed but not used in the gaps. Demographic parity is shown for context, not optimized."
    )
    fthr = st.slider("Threshold", 0.02, 0.60, float(threshold), 0.01, format="%.2f",
                     help="Defaults to the deployed threshold. Move it to see how the gaps respond.")
    tbl = group_table(oof[prot].fillna("missing"), oof.y, oof.p, fthr)
    gp = gaps(tbl)
    q = st.columns(3)
    q[0].metric("Equalized-odds gap (primary)", f"{gp['equalized_odds_gap']:.1%}")
    q[1].metric("Approval gap among repayers", f"{gp['gap_approve_given_repay']:.1%}")
    q[2].metric("Demographic-parity gap (context)", f"{gp['demographic_parity_gap']:.1%}")

    big = tbl[tbl.n >= MIN_GROUP]
    measures = {
        "approval_rate": "approved (all)",
        "approve_given_repay": "approved | would repay",
        "approve_given_default": "approved | would default",
    }
    f = go.Figure()
    for colour, (grp, r_) in zip([BLUE, ORANGE], big.iterrows()):
        f.add_bar(
            x=list(measures.values()), y=[r_[c] for c in measures], name=f"{grp} (n={r_.n:,})",
            marker_color=colour, text=[f"{r_[c]:.1%}" for c in measures], textposition="outside",
            hovertemplate="%{y:.1%}<extra></extra>",
        )
    f.update_yaxes(tickformat=".0%", range=[0, 1.15])
    f.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, title="Rates by group")
    plot(f, 360)
    st.dataframe(
        tbl.style.format({
            "default_rate": "{:.1%}", "approval_rate": "{:.1%}",
            "approve_given_repay": "{:.1%}", "approve_given_default": "{:.1%}",
        }),
        width="stretch",
    )

    fair = metrics.get("fairness")
    if fair and "ablation" in fair:
        action = "adding" if fair["ablation"]["protected_in_model"] else "removing"
        st.markdown(
            f"**Ablation: does {action} CODE_GENDER as a model input move the gap?** "
            "(deployed threshold, training OOF)"
        )
        comp = pd.DataFrame(
            {
                "main model": [fair["main"]["equalized_odds_gap"], fair["main"]["demographic_parity_gap"]],
                "ablation": [fair["ablation"]["equalized_odds_gap"], fair["ablation"]["demographic_parity_gap"]],
            },
            index=["equalized-odds gap", "demographic-parity gap"],
        )
        f = go.Figure()
        for colour, col in zip([BLUE, ORANGE], comp.columns):
            f.add_bar(
                x=comp.index, y=comp[col], name=col, marker_color=colour,
                text=[f"{v:.1%}" for v in comp[col]], textposition="outside",
                hovertemplate="%{y:.2%}<extra></extra>",
            )
        f.update_yaxes(tickformat=".0%", rangemode="tozero")
        f.update_layout(barmode="group", bargap=0.35)
        plot(f, 320)
        if "proxy_auc_gender_from_other_features" in fair:
            st.metric(
                "How well other features predict gender (AUC)",
                f"{fair['proxy_auc_gender_from_other_features']:.3f}",
                help="0.5 = no proxy. Well above 0.5 means removing CODE_GENDER alone cannot remove its effect.",
            )
    else:
        st.info("Ablation results appear here after `python -m src.fairness` has run.")
