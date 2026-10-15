"""Streamlit demo: score an applicant and explore the decisions behind the model.

Four tabs:
    Decide      - form -> calibrated default probability, APPROVE/DENY, expected
                  profit (with fixed cost), denial reasons, and a what-if chart
                  over loan size.
    Model       - the 5-model PR-AUC ladder, test-set metrics (read from
                  metrics.json, never recomputed), reliability and tree figures.
    Profit lab  - move LGD / interest / fixed cost and watch the profit curve and
                  optimal threshold move, on TRAINING out-of-fold predictions.
    Fairness    - approval rates by CODE_GENDER at any threshold (training
                  out-of-fold), plus the saved ablation from fairness.py.

Why it is built this way:
    - Loads ONLY artifacts/calibrated_pipeline.pkl for scoring, never the
      uncalibrated pipeline.pkl: the profit rule needs true probabilities.
    - Never loads the test set. Test numbers are read from metrics.json, where
      evaluate.py wrote them on its single test pass. Everything interactive runs
      on oof_predictions.csv (training split), so playing with sliders cannot
      leak into the reported result.
    - Form fields the user doesn't set take the training median (numeric) or
      most common value (categorical). An applicant with ~110 missing fields is
      unlike anyone the model saw and gets a misleading score.
    - Categorical columns come from the fitted pipeline itself (the columns its
      dtype selector picked at fit time), so the app can't drift from training.

What breaks if an assumption is wrong:
    - If metrics.json or the .pkl files are from an older run than
      oof_predictions.csv, the deployed threshold and the curves disagree.
      Re-run train -> evaluate -> fairness together.
    - The profit lab's optimum is an exploration aid. The deployed threshold is
      the one evaluate.py chose from config.yaml; changing it means changing
      config.yaml and re-running, not moving a slider.

Run:  streamlit run app.py
"""
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import artifact_path, load_config, load_train, read_metrics
from src.evaluate import breakeven_threshold
from src.fairness import MIN_GROUP, adverse_action_reasons, gaps, group_table

# Reference data-viz palette (validated categorical order; status colours reserved).
BLUE, ORANGE, MUTED, AXIS = "#2a78d6", "#eb6834", "#898781", "#c3c2b7"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
GRID = np.linspace(0.01, 0.99, 197)   # same threshold grid as evaluate.py

st.set_page_config(page_title="Loan Risk Studio", page_icon="🏦", layout="wide")
cfg = load_config()
cost = cfg["cost_assumptions"]
id_col, prot = cfg["data"]["id_column"], cfg["data"]["protected_attribute"]


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return joblib.load(artifact_path(cfg, "calibrated_pipeline.pkl"))


@st.cache_resource(show_spinner="Loading training data (first run only)…")
def load_training_view():
    """Train-split OOF predictions joined to loan size and gender, plus a typical applicant."""
    X, _ = load_train(cfg)
    oof = pd.read_csv(artifact_path(cfg, "oof_predictions.csv"))
    oof = oof.merge(X[[id_col, cost["exposure_column"], prot]], on=id_col, how="left")
    typical = {c: (X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else X[c].mode().iloc[0])
               for c in X.columns}
    return oof, typical


def categorical_columns(model):
    """The categorical inputs the fitted pipeline actually used (picked by dtype at fit)."""
    prep = model.calibrated_classifiers_[0].estimator["features"]["prep"]
    return set(next(cols for name, _, cols in prep.transformers_ if name == "cat"))


def plot(fig, height=320):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
    fig.update_xaxes(showgrid=False, linecolor=AXIS)
    fig.update_yaxes(gridcolor="rgba(137,135,129,0.2)", zeroline=False)
    st.plotly_chart(fig, width="stretch", theme="streamlit")


def decision_card(approved, p, threshold):
    colour, icon, label = (GOOD, "✅", "APPROVE") if approved else (CRITICAL, "⛔", "DENY")
    rel = "below" if approved else "at or above"
    st.markdown(
        f"""<div style="border-left:6px solid {colour};padding:14px 18px;border-radius:8px;
        background:rgba(137,135,129,0.08)">
        <div style="font-size:1.6rem;font-weight:700">{icon} {label}</div>
        <div style="opacity:.8">Default risk {p:.1%} is {rel} the approval threshold {threshold:.1%}.</div>
        </div>""", unsafe_allow_html=True)


def profit_curve(y, p, exposure, lgd, interest, fixed):
    """Profit at every threshold in GRID in one pass (sort by p, cumulative sum)."""
    order = np.argsort(p)
    gain = (np.where(y == 0, interest * exposure, -lgd * exposure) - fixed)[order]
    csum = np.concatenate([[0.0], np.cumsum(gain)])
    n_approved = np.searchsorted(p[order], GRID, side="left")
    return csum[n_approved], n_approved / len(p)


# ---------------------------------------------------------------- guards
metrics = read_metrics(cfg)
if "threshold" not in metrics or "test" not in metrics:
    st.error("No threshold or test results found. Run `python -m src.train` then `python -m src.evaluate` first.")
    st.stop()
threshold = metrics["threshold"]["value"]
lgd, interest, fixed = cost["loss_given_default"], cost["expected_interest"], cost["fixed_cost_per_loan"] or 0
model = load_model()
cats = categorical_columns(model)
oof, typical = load_training_view()

# ---------------------------------------------------------------- header
st.title("🏦 Loan Risk Studio")
st.caption("Home Credit default-risk model · calibrated LightGBM · expected-profit decision rule")
h = st.columns(4)
h[0].metric("Approval threshold", f"{threshold:.1%}", help="Approve if calibrated P(default) is below this. "
            "Chosen on training out-of-fold predictions.")
h[1].metric("Loss given default", f"{lgd:.0%}")
h[2].metric("Interest earned if repaid", f"{interest:.0%}")
h[3].metric("Fixed cost per loan", f"{fixed:,.0f}")

tab_decide, tab_model, tab_profit, tab_fair = st.tabs(["🧮 Decide", "📊 Model", "💰 Profit lab", "⚖️ Fairness"])

# ================================================================= Decide
with tab_decide:
    form, result = st.columns([1, 1.25], gap="large")
    with form:
        st.subheader("Applicant")
        c1, c2 = st.columns(2)
        income = c1.number_input("Annual income", 10_000, 10_000_000, int(typical["AMT_INCOME_TOTAL"]), step=5_000)
        credit = c1.number_input("Loan amount", 45_000, 4_050_000, int(typical["AMT_CREDIT"]), step=10_000)
        annuity = c1.number_input("Yearly repayment (annuity)", 1_000, 300_000, int(typical["AMT_ANNUITY"]),
                                  step=1_000)
        age = c2.slider("Age (years)", 20, 70, int(-typical["DAYS_BIRTH"] / 365))
        no_job = c2.checkbox("Pensioner / not employed")
        employed = c2.slider("Years employed", 0, 45, 5, disabled=no_job)
        education = c2.selectbox("Education", ["Secondary / secondary special", "Higher education",
                                               "Incomplete higher", "Lower secondary", "Academic degree"])
        st.markdown("**External credit scores** (0 = worst, 1 = best; the strongest signals in the data)")
        e = st.columns(3)
        ext = [e[k].slider(f"Score {k + 1}", 0.0, 1.0, round(float(typical[f"EXT_SOURCE_{k + 1}"]), 2), 0.01)
               for k in range(3)]

    def applicant_rows(loans):
        """One row per loan size, holding everything else fixed (annuity scales with the loan)."""
        rows = pd.DataFrame([typical] * len(loans))
        rows["AMT_INCOME_TOTAL"] = income
        rows["AMT_CREDIT"] = loans
        rows["AMT_GOODS_PRICE"] = loans          # assume the loan finances the goods in full
        rows["AMT_ANNUITY"] = annuity * np.asarray(loans) / credit
        rows["DAYS_BIRTH"] = -age * 365
        rows["DAYS_EMPLOYED"] = 365243 if no_job else -employed * 365   # 365243 = dataset's "not employed" code
        rows["NAME_EDUCATION_TYPE"] = education
        for k in range(3):
            rows[f"EXT_SOURCE_{k + 1}"] = ext[k]
        cols = list(model.feature_names_in_)
        rows = rows[cols]
        # numeric inputs as float; text columns (incl. ones the model drops, e.g. CODE_GENDER) left as-is
        return rows.astype({c: "float64" for c in cols
                            if c not in cats and isinstance(typical[c], (int, float, np.number))})

    row = applicant_rows([credit])
    p = float(model.predict_proba(row)[0, 1])
    approved = p < threshold
    expected = credit * ((1 - p) * interest - p * lgd) - fixed

    with result:
        st.subheader("Decision")
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Calibrated probability of default"},
            gauge={"axis": {"range": [0, max(50, p * 100 + 5)], "ticksuffix": "%"},
                   "bar": {"color": CRITICAL if not approved else GOOD, "thickness": 0.3},
                   "steps": [{"range": [0, threshold * 100], "color": "rgba(12,163,12,0.15)"},
                             {"range": [threshold * 100, 100], "color": "rgba(208,59,59,0.12)"}],
                   "threshold": {"line": {"color": MUTED, "width": 3}, "value": threshold * 100}}))
        g.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=0))
        st.plotly_chart(g, width="stretch", theme="streamlit")
        decision_card(approved, p, threshold)
        m = st.columns(2)
        m[0].metric("Expected profit if approved", f"{expected:,.0f}",
                    delta="profitable" if expected > 0 else "loss-making",
                    delta_color="normal" if expected > 0 else "inverse")
        m[1].metric("Break-even risk for this loan", f"{max(breakeven_threshold(credit, lgd, interest, fixed), 0):.1%}",
                    help="Risk at which expected profit is exactly zero for this loan size.")
        if not approved:
            st.markdown("**Main reasons for denial** (features pushing risk up most)")
            fold_model = model.calibrated_classifiers_[0].estimator
            for reason in adverse_action_reasons(fold_model, row)[0]:
                name, _, value = reason.partition(" = ")
                try:
                    value = f"{float(value):,.4g}"
                except ValueError:
                    pass
                st.markdown(f"- `{name}` = {value}")

    st.divider()
    st.subheader("What if the loan were a different size?")
    st.caption("Same applicant, loan amount varied; repayment scales with the loan.")
    loans = np.geomspace(45_000, 4_000_000, 60)
    ps = model.predict_proba(applicant_rows(loans))[:, 1]
    w1, w2 = st.columns(2)
    with w1:
        f = go.Figure()
        f.add_scatter(x=loans, y=ps, name="this applicant's risk", line=dict(color=BLUE, width=2),
                      hovertemplate="%{y:.1%}")
        f.add_hline(y=threshold, line=dict(color=MUTED, dash="dash"), annotation_text="approval threshold")
        f.add_scatter(x=[credit], y=[p], mode="markers", name="current loan",
                      marker=dict(size=11, color=BLUE, line=dict(width=2, color="white")), hovertemplate="%{y:.1%}")
        f.update_xaxes(type="log", title="loan amount")
        f.update_yaxes(tickformat=".0%", title="P(default)")
        f.update_layout(title="Default risk vs loan size")
        plot(f)
    with w2:
        prof = loans * ((1 - ps) * interest - ps * lgd) - fixed
        f = go.Figure()
        f.add_scatter(x=loans, y=prof, name="expected profit", line=dict(color=BLUE, width=2),
                      hovertemplate="%{y:,.0f}")
        f.add_hline(y=0, line=dict(color=AXIS))
        f.add_scatter(x=[credit], y=[expected], mode="markers", name="current loan",
                      marker=dict(size=11, color=BLUE, line=dict(width=2, color="white")), hovertemplate="%{y:,.0f}")
        f.update_xaxes(type="log", title="loan amount")
        f.update_yaxes(title="expected profit")
        f.update_layout(title="Expected profit vs loan size", showlegend=False)
        plot(f)

# ================================================================= Model
with tab_model:
    test = metrics["test"]
    st.subheader("Held-out test set (evaluated once)")
    t = st.columns(4)
    t[0].metric("PR-AUC (primary)", f"{test['calibrated']['pr_auc']:.3f}",
                help=f"Base rate is {oof.y.mean():.1%}; a random model scores about that.")
    t[1].metric("ROC-AUC", f"{test['calibrated']['roc_auc']:.3f}")
    t[2].metric("Brier score", f"{test['calibrated']['brier_score']:.4f}",
                delta=f"{test['calibrated']['brier_score'] - test['uncalibrated']['brier_score']:+.4f} vs uncalibrated",
                delta_color="inverse")
    t[3].metric("Approval rate", f"{test['approval_rate']:.1%}")
    uplift = test["profit_at_threshold"] - test["profit_approve_all"]
    st.info(f"On the test set the threshold earns **{test['profit_at_threshold']:,.0f}**, versus "
            f"**{test['profit_approve_all']:,.0f}** for approving everyone: **{uplift:+,.0f}**.")

    l, r = st.columns(2)
    with l:
        cv = pd.DataFrame(metrics["cv"]).T.reset_index(names="model").sort_values("pr_auc")
        f = go.Figure(go.Bar(x=cv.pr_auc, y=cv.model, orientation="h",
                             marker_color=[BLUE if mdl == "lightgbm" else AXIS for mdl in cv.model],
                             text=[f"{v:.3f}" for v in cv.pr_auc], textposition="outside",
                             hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        f.add_vline(x=oof.y.mean(), line=dict(color=MUTED, dash="dot"), annotation_text="base rate")
        f.update_layout(title="5-model ladder: 5-fold CV PR-AUC (training split)", hovermode="closest")
        f.update_xaxes(range=[0, cv.pr_auc.max() * 1.2])
        plot(f, 340)
        if "accuracy" in metrics["cv"].get("baseline", {}):
            st.caption(f"Why not accuracy: predicting 'nobody defaults' already scores "
                       f"{metrics['cv']['baseline']['accuracy']:.1%} accuracy, with zero useful predictions.")
    with r:
        st.markdown("**Calibration: before vs after isotonic**")
        st.image(str(artifact_path(cfg, "figures/reliability.png")), width="stretch")
    with st.expander("Printable depth-4 decision tree"):
        st.image(str(artifact_path(cfg, "figures/decision_tree.png")), width="stretch")

# ================================================================= Profit lab
with tab_profit:
    st.subheader("Profit lab")
    st.caption("Explore cost assumptions on **training out-of-fold** predictions. The test set is not used here. "
               "The deployed threshold only changes when config.yaml changes and evaluate.py is re-run.")
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
        f.add_scatter(x=GRID, y=curve, name="profit", line=dict(color=BLUE, width=2), hovertemplate="%{y:,.0f}")
        f.add_vline(x=GRID[best], line=dict(color=BLUE, dash="dash"), annotation_text="best")
        f.add_vline(x=threshold, line=dict(color=MUTED, dash="dot"), annotation_text="deployed",
                    annotation_position="bottom right")
        f.update_xaxes(title="threshold (approve if P(default) < t)", tickformat=".0%")
        f.update_yaxes(title="total profit (training OOF)")
        f.update_layout(title="Profit vs threshold", showlegend=False)
        plot(f)
    with b:
        f = go.Figure()
        f.add_scatter(x=GRID, y=approval, name="approval rate", line=dict(color=BLUE, width=2),
                      hovertemplate="%{y:.1%}")
        f.add_vline(x=GRID[best], line=dict(color=BLUE, dash="dash"))
        f.update_xaxes(title="threshold", tickformat=".0%")
        f.update_yaxes(title="share approved", tickformat=".0%", range=[0, 1.02])
        f.update_layout(title="Approval rate vs threshold", showlegend=False)
        plot(f)

    st.markdown("**Break-even threshold by loan size** — with a fixed cost, small loans need lower risk to pay off")
    sizes = np.geomspace(max(lab_fix / lab_int * 0.5, 20_000) if lab_fix else 20_000, 4_000_000, 200)
    f = go.Figure()
    f.add_scatter(x=sizes, y=np.clip(breakeven_threshold(sizes, lab_lgd, lab_int, lab_fix), 0, None),
                  name="break-even threshold", line=dict(color=BLUE, width=2), hovertemplate="%{y:.1%}")
    f.add_hline(y=GRID[best], line=dict(color=MUTED, dash="dash"), annotation_text="best single threshold")
    lo, hi = np.percentile(ex_o, [5, 95])
    f.add_vrect(x0=lo, x1=hi, fillcolor=MUTED, opacity=0.1, line_width=0,
                annotation_text="middle 90% of loans", annotation_position="top left")
    f.update_xaxes(type="log", title="loan amount")
    f.update_yaxes(tickformat=".0%", title="approve if P(default) <", rangemode="tozero")
    f.update_layout(showlegend=False)
    plot(f, 300)

# ================================================================= Fairness
with tab_fair:
    st.subheader("Fairness by CODE_GENDER")
    st.caption(f"Primary metric: **equalized odds** — repayers should be approved at the same rate in every "
               f"group, and so should defaulters. Computed on training out-of-fold predictions. "
               f"Groups under {MIN_GROUP} people are listed but not used in the gaps.")
    fthr = st.slider("Threshold", 0.02, 0.60, float(threshold), 0.01, format="%.2f",
                     help="Defaults to the deployed threshold. Move it to see how the gaps respond.")
    tbl = group_table(oof[prot].fillna("missing"), oof.y, oof.p, fthr)
    gp = gaps(tbl)
    q = st.columns(3)
    q[0].metric("Equalized-odds gap (primary)", f"{gp['equalized_odds_gap']:.1%}")
    q[1].metric("Approval gap among repayers", f"{gp['gap_approve_given_repay']:.1%}")
    q[2].metric("Demographic-parity gap (context)", f"{gp['demographic_parity_gap']:.1%}")

    big = tbl[tbl.n >= MIN_GROUP]
    measures = {"approval_rate": "approved (all)", "approve_given_repay": "approved | would repay",
                "approve_given_default": "approved | would default"}
    f = go.Figure()
    for colour, (grp, r_) in zip([BLUE, ORANGE], big.iterrows()):
        f.add_bar(x=list(measures.values()), y=[r_[c] for c in measures], name=f"{grp} (n={r_.n:,})",
                  marker_color=colour, text=[f"{r_[c]:.1%}" for c in measures], textposition="outside",
                  hovertemplate="%{y:.1%}")
    f.update_yaxes(tickformat=".0%", range=[0, 1.1])
    f.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, title="Rates by group")
    plot(f, 360)
    st.dataframe(tbl.style.format({"default_rate": "{:.1%}", "approval_rate": "{:.1%}",
                                   "approve_given_repay": "{:.1%}", "approve_given_default": "{:.1%}"}),
                 width="stretch")

    fair = metrics.get("fairness")
    if fair and "ablation" in fair:
        st.markdown(f"**Ablation: does {'adding' if fair['ablation']['protected_in_model'] else 'removing'} "
                    f"CODE_GENDER as a model input move the gap?** (deployed threshold)")
        comp = pd.DataFrame({"main model": [fair["main"]["equalized_odds_gap"], fair["main"]["demographic_parity_gap"]],
                             "ablation": [fair["ablation"]["equalized_odds_gap"],
                                          fair["ablation"]["demographic_parity_gap"]]},
                            index=["equalized-odds gap", "demographic-parity gap"])
        f = go.Figure()
        for colour, col in zip([BLUE, ORANGE], comp.columns):
            f.add_bar(x=comp.index, y=comp[col], name=col, marker_color=colour,
                      text=[f"{v:.1%}" for v in comp[col]], textposition="outside", hovertemplate="%{y:.2%}")
        f.update_yaxes(tickformat=".0%", rangemode="tozero")
        f.update_layout(barmode="group", bargap=0.35)
        plot(f, 320)
        if "proxy_auc_gender_from_other_features" in fair:
            st.metric("How well other features predict gender (AUC)",
                      f"{fair['proxy_auc_gender_from_other_features']:.3f}",
                      help="0.5 = no proxy. Well above 0.5 means removing CODE_GENDER alone can't remove its effect.")
    else:
        st.info("Ablation results appear here after `python -m src.fairness` has run.")
