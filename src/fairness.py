"""Group fairness metrics, protected-attribute ablation, and SHAP adverse-action reasons.

Primary fairness metric and tradeoff:
    TODO (team): state which one is primary, demographic parity or equalized odds,
    and argue why. They are mathematically incompatible when default base rates
    differ between CODE_GENDER groups, so this module satisfies exactly one.
    Write down what the chosen metric gives up.

Owns:
    - Group metrics on the calibrated pipeline's decisions, split by
      CODE_GENDER (check the actual values on load; don't assume M/F/XNA is
      the full set).
    - The proxy ablation: drop CODE_GENDER, retrain, and report whether the gap
      moves. If it doesn't, other features are acting as proxies.
    - SHAP-based adverse-action reasons for individual denials.

Does NOT own:
    - Model fitting, except the ablation retrain, which reuses train.py's code
      path instead of reimplementing it.
    - Calibration or threshold selection (evaluate.py).

Reads:  config.yaml -> fairness, artifacts/pipeline.pkl, data/processed/
Writes: artifacts/metrics.json (fairness section), artifacts/figures/

What breaks if an assumption is wrong:
    - If a rare CODE_GENDER value (e.g. XNA) is silently dropped or merged,
      the group metrics describe a population that doesn't match the data.
    - If metrics use a different threshold from evaluate.py, the fairness gap
      describes decisions the model never actually makes.
"""
