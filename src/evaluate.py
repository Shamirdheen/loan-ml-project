"""Calibrate the model, compute metrics, sweep the profit threshold, and touch the test set once.

Owns:
    - Calibrating LightGBM with CalibratedClassifierCV (isotonic, split mode from
      config.yaml), fit on TRAINING data only. Reliability curves before and after.
    - PR-AUC (primary), ROC-AUC, and Brier score. Accuracy is reported once, in
      the baseline row, to show why it was abandoned at an 8% base rate.
    - The expected-profit threshold sweep, using loss_given_default and
      expected_interest from config.yaml. Must fail loudly if either is null;
      never fall back to a default.
    - The single, final evaluation on the held-out test set.

Does NOT own:
    - Fairness metrics (fairness.py).
    - Model fitting beyond calibration (train.py).

Reads:  config.yaml, artifacts/pipeline.pkl, data/processed/ (frozen split)
Writes: artifacts/metrics.json, artifacts/figures/

What breaks if an assumption is wrong:
    - If calibration is fit on test rows, the "touch test exactly once" rule is
      broken and the reported numbers are no longer an honest estimate.
    - If uncalibrated probabilities reach the profit code, the chosen threshold
      optimizes against distorted risk estimates and the profit number is wrong.
"""
