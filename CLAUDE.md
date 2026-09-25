# Loan Approval Risk Model — Project Context

## What this is
A tabular classification project (Home Credit `application_train.csv`) built
for a college ML course. The dataset and models are deliberately standard —
this project earns its grade from four things that are NOT standard:

1. An expected-profit threshold, not accuracy or 0.5.
2. A calibrated probability output (LightGBM's raw probabilities are distorted
   by class weighting; calibration is mandatory, not optional).
3. A stated fairness metric with an explicit tradeoff argument.
4. A written hypothesis before every feature or experiment, so nothing looks
   copy-pasted from a tutorial.

Team of 5, one module each (see folder structure). One-month timeline, weekly
faculty demo checkpoints.

## Non-negotiable rules for any code written here

- **No leakage.** Every transform (impute, scale, encode) lives inside a
  single `sklearn.Pipeline`, fit only on training folds. Never call `.fit()`
  on anything before the train/test split.
- **No SMOTE by default.** Use `class_weight='balanced'` /
  `scale_pos_weight`. If SMOTE is added, it's an explicit ablation with a
  calibration-damage comparison, never the default path.
- **No neural networks.** Gradient-boosted trees win on this data class. If a
  NN is required by the rubric, it's one clearly-labeled losing baseline, not
  the main model.
- **Max 5 models**: majority-class baseline, logistic regression, decision
  tree (depth 4, for the printable figure), random forest, LightGBM. Do not
  add XGBoost/CatBoost — no marginal benefit for the time cost.
- **Calibrate LightGBM's output** (`CalibratedClassifierCV`, isotonic) before
  it touches the profit-threshold code. Plot reliability curves before/after.
- **Touch the held-out test set exactly once**, at the very end. Everything
  before that is stratified 5-fold CV on the training split.
- **Primary metric is PR-AUC**, not accuracy or plain ROC-AUC (8% base rate
  makes both misleading). Report accuracy once, in the baseline row, to show
  why it was abandoned.
- **No algorithm-internals math** (no sigmoid derivations, no Gini formula,
  no gradient boosting loss). The only formulas that belong anywhere in this
  project are the ones for decisions the team made itself: expected profit,
  the threshold rule, the fairness metric.

## Style

- Simple beats clever. No over-engineered abstractions for a 4-week project.
- Single responsibility per file (see structure below) — don't let training
  logic leak into `evaluate.py` or vice versa.
- Before generating a chunk of modeling code, state the hypothesis being
  tested in a comment first. A cell with no stated "why" before it shouldn't
  exist.
- When handing back code, always explain: what it does, why it's structured
  that way, and what breaks if an assumption behind it is wrong.
- Work incrementally. Scaffold, then one module at a time. Don't generate the
  whole pipeline in one pass — stop after each module for review.

## Folder structure

```
loan-approval/
  config.yaml              # paths, columns, cost assumptions, seed
  data/
    raw/                   # application_train.csv — never written to
    processed/              # generated, gitignored
  src/
    data.py                 # load + stratified split
    features.py              # engineered features + preprocessing pipelines
                              #   (LR/RF branch, LightGBM native-categorical branch)
    model.py                  # the 5-model ladder, defined once
    train.py                   # CV fit, save pipeline artifact
    evaluate.py                 # PR-AUC/ROC-AUC/Brier, calibration, profit-threshold sweep
    fairness.py                  # group metrics, proxy ablation, SHAP adverse-action reasons
  notebooks/
    01_eda.ipynb
  artifacts/
    pipeline.pkl
    metrics.json
    figures/
  app.py                        # Streamlit demo — week 4 only, do not build early
```

## Dataset

Home Credit Default Risk, `application_train.csv` only (ignore the 6
auxiliary tables — not worth the join time on this timeline). ~307k rows,
~122 columns, ~8% positive (default) rate. Target column: `TARGET`.
Protected attribute for fairness work: `CODE_GENDER` (verify exact
values/encoding on load — don't assume XNA/M/F is the full set).

## Cost assumptions (fill in before evaluate.py can run)

- `loss_given_default`: team decides, document reasoning in `config.yaml`
- `expected_interest`: team decides, document reasoning in `config.yaml`

These numbers are the actual grade-earning decision in this project. Do not
default them without writing down why.

## Fairness

Pick ONE of demographic parity / equalized odds as primary — they are
mathematically incompatible when base rates differ between groups, so do not
try to satisfy both. State the tradeoff explicitly in `fairness.py`'s module
docstring. Run the `CODE_GENDER`-dropped ablation and report whether the gap
moves.
