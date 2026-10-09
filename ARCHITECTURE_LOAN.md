# Architecture — Loan Approval Risk Model

This is the technical spec behind `CLAUDE.md`'s folder structure. Read this
before touching `src/`.

## Core design decision: one Pipeline object, not two code paths

Preprocessing and model are wrapped in a single `sklearn.Pipeline`, fit as
one unit, saved as one `.pkl`.

**Why this is the decision that matters most:** the classic failure on this
exact dataset is fitting an imputer or scaler on the full data before
splitting — median income gets computed using test rows, validation numbers
look better than they are, and nobody notices until the demo. Wrapping
everything in one `Pipeline` and passing that object to `cross_val_score`
makes leakage structurally impossible — the transforms refit inside every
fold automatically.

It also removes train/serve skew. Whatever `app.py` loads at the end is
byte-identical to what was evaluated. There is no second code path that
could silently drift from the first.

## Data flow

```
data/raw/application_train.csv
        │
        ▼
   data.py            load, stratified train/test split (test set frozen here)
        │
        ▼
   features.py         define preprocessing Pipeline (two branches — see below)
        │                does NOT fit anything yet
        ▼
   model.py             defines the 5-model ladder (estimator objects only)
        │
        ▼
   train.py              combines features.py's Pipeline + model.py's estimator
        │                 into one Pipeline, fits via 5-fold CV on train split,
        │                 saves artifacts/pipeline.pkl
        ▼
   evaluate.py            loads pipeline.pkl, calibrates (CalibratedClassifierCV)
        │                  on the train split, saves calibrated_pipeline.pkl,
        │                  picks the profit threshold on train out-of-fold
        │                  probabilities, then touches the held-out test set
        │                  exactly once to report at that fixed threshold
        ▼
   fairness.py             loads calibrated_pipeline.pkl + the chosen threshold,
                            computes group metrics, runs the protected-attribute
                            ablation, generates SHAP-based adverse-action reasons
```

`app.py` (week 4 only) loads `artifacts/calibrated_pipeline.pkl` directly — no
reimplementation of preprocessing or calibration, because the artifact
already contains both. It must NOT load the uncalibrated `pipeline.pkl`.

## Two preprocessing branches inside features.py

- **Branch A (logistic regression, decision tree, random forest):** one-hot encode
  categoricals, median-impute numerics, standard-scale.
- **Branch B (LightGBM):** native categorical handling, raw nulls passed
  through — LightGBM has a learned default direction for missing values, and
  imputing before it sees the data throws that away.

Both branches are `ColumnTransformer`s defined in the same file because they
share the same column-type lists (categorical vs numeric vs ID columns to
drop). Keeping that list in one place means a schema change updates both
branches at once instead of two files quietly drifting apart.

## Why train.py and evaluate.py are separate files

Training produces probabilities. The approval decision comes from
thresholding those probabilities against expected profit, which needs the
loss-given-default and interest assumptions from `config.yaml` —
`train.py` doesn't need either. Splitting them means you can sweep
thresholds, change the cost assumptions, or swap the fairness metric without
retraining anything. That's a fifteen-second iteration loop instead of a
five-minute one, which matters when adjusting numbers live in front of
faculty.

## Module responsibilities

| File | Owns | Does NOT own |
|---|---|---|
| `data.py` | loading, stratified split, freezing the test set | any transform of feature values |
| `features.py` | preprocessing Pipelines (both branches), engineered features + their stated hypotheses | fitting — only defines, `train.py` fits |
| `model.py` | the 5 estimator definitions from `config.yaml` | preprocessing, evaluation |
| `train.py` | CV fit, artifact saving | metric computation beyond what CV needs for model selection |
| `evaluate.py` | calibration, PR-AUC/ROC-AUC/Brier, profit-threshold sweep, the one-time test-set touch | fairness metrics |
| `fairness.py` | group metrics, proxy ablation, SHAP adverse-action reasons | model fitting |

## Reusable vs. project-specific

`data.py`, `train.py`, `evaluate.py` are generic tabular-classification
scaffolding — they'd work on any CSV with a binary target. `features.py` and
`fairness.py` are loan-specific: which columns are categorical, which
attribute is protected, what counts as an adverse-action reason. Keeping the
domain knowledge concentrated in those two files means the rest of the
pipeline stays swappable.

## What breaks if requirements shift

- **Different dataset:** touch `features.py` and `config.yaml`. Nothing else.
- **Different model** (e.g., swap random forest for any other
  sklearn-compatible estimator): one entry in `config.yaml`, read by `model.py`. The Pipeline interface is unchanged, so `train.py` and
  `evaluate.py` don't notice.
- **Multi-class or loan-amount regression instead of binary approval:**
  `model.py` and the metrics half of `evaluate.py` change. `data.py` and
  `features.py` survive untouched.
- **"Show it works on a new applicant" demo request:** that's `app.py`
  only, and it's trivial because `pipeline.pkl` is self-contained — no
  preprocessing logic needs to be reimplemented in the app.

## Team split (5 people, 5 modules)

`data.py` + `notebooks/01_eda.ipynb` · `features.py` · `train.py` +
`model.py` · `evaluate.py` (profit curve + calibration) · `fairness.py`
(SHAP + proxy ablation). They integrate through `config.yaml`, the frozen
`data/processed/split.csv`, and the saved `.pkl` artifacts, so no module
blocks another mid-week. Artifacts are gitignored. Each teammate regenerates
them locally (`python -m src.data`, then `python -m src.train`), and the
fixed seed makes that reproducible. `split.csv` is the one generated file
that IS committed, so everyone shares the identical test set. Put the least
confident team member on the EDA notebook — most visible piece in week one,
hardest to get structurally wrong.
