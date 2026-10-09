"""Define the preprocessing Pipelines and engineered features. Nothing is fit here.

Owns:
    - The single source of truth for which columns are categorical, numeric, or
      dropped (ID, target, protected attribute). Types are detected by dtype at
      fit time, and both branches share the same selectors.
    - Branch A (logistic regression, decision tree, random forest): a
      ColumnTransformer that one-hot encodes categoricals
      (handle_unknown='ignore', so a category seen in only one CV fold doesn't
      crash), median-imputes numerics, and standard-scales them.
    - Branch B (LightGBM): a ColumnTransformer that uses native categorical
      handling and passes raw nulls through. LightGBM learns a default direction
      for missing values, and imputing first would throw that away. It needs
      set_output(transform="pandas") plus a cast to 'category' dtype. By default
      ColumnTransformer returns numpy, which silently turns categoricals into
      plain numbers.
    - Whether CODE_GENDER is a model input (config.yaml ->
      fairness.protected_in_model). Refuse to build while that is null.
    - Engineered features, each with a written hypothesis stated before it.
      Known data quirk to handle here: DAYS_EMPLOYED == 365243 is a placeholder
      code (mostly pensioners/unemployed), not a real value. Flag it in EDA and
      handle it explicitly, with a stated hypothesis.

Does NOT own:
    - Fitting. This module only defines objects; train.py fits them inside CV.

Reads:  config.yaml (column names, protected attribute)
Writes: nothing

What breaks if an assumption is wrong:
    - If an engineered feature is computed outside the Pipeline using statistics
      from the whole dataset (e.g. a group mean), it leaks test information.
    - If the column lists drift from the actual CSV schema, ColumnTransformer
      fails loudly, or worse, silently drops columns (remainder='drop').
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Column types are detected by dtype when the pipeline is FIT (inside each CV fold),
# not hardcoded: any non-numeric column is categorical, everything else is numeric.
# Tradeoff: a column that is numeric in the CSV but categorical in meaning is treated
# as numeric, and a surprise text column is silently one-hot encoded instead of
# failing loudly. EDA section 2 is where to eyeball the detected list.
DAYS_EMPLOYED_PLACEHOLDER = 365243


def add_engineered_features(X):
    """Row-wise features only. No column statistics are used, so this cannot leak.

    Hypotheses (stated up front; not yet validated by an ablation):
      DAYS_EMPLOYED placeholder: 365243 (~1000 years) marks pensioners/unemployed.
        Left as a number it looks like "employed forever" and warps scaling and
        tree splits. Set to NaN and keep a flag: the group itself may default
        at a different rate than people with a real employment length.
      CREDIT_INCOME_RATIO: a loan that is large relative to income is harder to
        repay than the raw amounts suggest -> higher default rate.
      ANNUITY_INCOME_RATIO: a big share of income going to repayment each period
        leaves less slack -> higher default rate.
      CREDIT_TERM: annuity / credit is the inverse of loan length; short, steep
        repayment schedules are riskier.
      EMPLOYED_AGE_RATIO: share of life spent employed; stability proxy.
      EXT_SOURCE_MEAN: the three external scores are the strongest known signals
        here; their mean is less noisy than any one, and tolerates one missing.
    """
    X = X.copy()
    emp = X["DAYS_EMPLOYED"]
    X["DAYS_EMPLOYED_ANOM"] = (emp == DAYS_EMPLOYED_PLACEHOLDER).astype(int)
    X["DAYS_EMPLOYED"] = emp.where(emp != DAYS_EMPLOYED_PLACEHOLDER)

    X["CREDIT_INCOME_RATIO"] = X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"]
    X["ANNUITY_INCOME_RATIO"] = X["AMT_ANNUITY"] / X["AMT_INCOME_TOTAL"]
    X["CREDIT_TERM"] = X["AMT_ANNUITY"] / X["AMT_CREDIT"]
    X["EMPLOYED_AGE_RATIO"] = X["DAYS_EMPLOYED"] / X["DAYS_BIRTH"]
    X["EXT_SOURCE_MEAN"] = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    # A zero denominator gives inf, which imputers and scalers choke on.
    return X.replace([np.inf, -np.inf], np.nan)


class _Selector:
    """Picks columns by dtype at fit time. A class (not a closure) so the fitted pipeline pickles.

    Excludes the ID, target and (unless allowed) the protected attribute.
    """

    def __init__(self, drop, categorical):
        self.drop = set(drop)
        self.categorical = categorical

    def __call__(self, df):
        return [c for c in df.columns
                if c not in self.drop and pd.api.types.is_numeric_dtype(df[c]) != self.categorical]


class _CategoryCaster(BaseEstimator, TransformerMixin):
    """Cast the non-numeric columns seen at fit to 'category' (for LightGBM's native handling).

    Remembering the columns at fit keeps predict-time casting identical even if a
    column happens to arrive all-missing (and so as float) in a single row.
    LightGBM remembers the categories seen at fit and maps unseen ones to missing.
    """

    def fit(self, X, y=None):
        self.columns_ = np.asarray(X.columns, dtype=object)
        self.cat_cols_ = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cat_cols_] = X[self.cat_cols_].astype("category")
        return X

    def get_feature_names_out(self, input_features=None):
        return self.columns_


def _columns(cfg, protected_in_model):
    """Return (categorical_selector, numeric_selector) after the ID/target/protected-attribute rules."""
    d = cfg["data"]
    drop = {d["id_column"], d["target_column"]}
    if not protected_in_model:
        drop.add(d["protected_attribute"])
    return _Selector(drop, categorical=True), _Selector(drop, categorical=False)


def build_features(cfg, branch, protected_in_model=None):
    """Unfitted preprocessing Pipeline. branch: "A" (LR/tree/RF) or "B" (LightGBM).

    protected_in_model overrides config.yaml only for the fairness ablation.
    Refuses to build while the config value is null (team decision).
    """
    if protected_in_model is None:
        protected_in_model = cfg["fairness"]["protected_in_model"]
    if protected_in_model is None:
        raise ValueError("fairness.protected_in_model is null in config.yaml; decide it first")

    cats, numeric = _columns(cfg, protected_in_model)
    engineer = FunctionTransformer(add_engineered_features).set_output(transform="pandas")

    if branch == "A":
        # Hypothesis: one-hot + median + scaling is what a linear model needs; the
        # tree models are indifferent to it, so one branch serves all three.
        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
                ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                  ("scale", StandardScaler())]), numeric),
            ],
            remainder="drop",
        )
        return Pipeline([("engineer", engineer), ("prep", pre)])

    if branch == "B":
        # Hypothesis: passing NaN straight through lets LightGBM learn "missing"
        # as a signal (EDA section 3); a median fill would erase it.
        pre = ColumnTransformer(
            [("cat", "passthrough", cats), ("num", "passthrough", numeric)],
            remainder="drop",
        ).set_output(transform="pandas")
        pre.verbose_feature_names_out = False
        return Pipeline([
            ("engineer", engineer),
            ("prep", pre),
            ("categorize", _CategoryCaster()),
        ]).set_output(transform="pandas")

    raise ValueError(f"branch must be 'A' or 'B', got {branch!r}")
