"""Combine preprocessing and model into one Pipeline, cross-validate, and save the artifact.

Owns:
    - Joining features.py's preprocessing branch with a model.py estimator into
      one sklearn.Pipeline (Branch A for LR/RF/tree, Branch B for LightGBM).
    - Stratified 5-fold CV on the TRAINING split only (cv_folds and seed from
      config.yaml). Transforms refit inside every fold, so leakage is structurally
      impossible.
    - Saving the chosen fitted Pipeline to artifacts/pipeline.pkl.

Does NOT own:
    - Metric computation beyond what CV needs for model selection (PR-AUC).
    - Calibration, profit thresholds, or cost assumptions (evaluate.py).
    - Any access to the held-out test set.

Reads:  config.yaml, data/processed/ (frozen split)
Writes: artifacts/pipeline.pkl

What breaks if an assumption is wrong:
    - If preprocessing is fit before being wrapped in the Pipeline, CV scores
      include leaked statistics and look better than they really are.
    - If the saved .pkl isn't the exact object that was evaluated, app.py
      serves a different model than the one reported (train/serve skew).
"""
