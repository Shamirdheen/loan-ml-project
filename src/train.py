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

import argparse

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree

from .data import artifact_path, load_config, load_train, update_metrics
from .features import build_features
from .model import BRANCH, build_models

MAIN_MODEL = "lightgbm"


def make_pipeline(cfg, name, y_train, protected_in_model=None):
    """features + model as ONE unfitted Pipeline (also used by fairness.py's ablation)."""
    model = build_models(cfg, y_train)[name]
    return Pipeline([
        ("features", build_features(cfg, BRANCH[name], protected_in_model)),
        ("model", model),
    ])


def cv_ladder(cfg, X, y):
    """Stratified 5-fold out-of-fold PR-AUC for each of the five models.

    Hypothesis: LightGBM > random forest > logistic regression > depth-4 tree >
    baseline on PR-AUC, with the baseline sitting at the base rate. If the order
    differs, the feature work or the class weighting is suspect.
    """
    skf = StratifiedKFold(cfg["data"]["cv_folds"], shuffle=True, random_state=cfg["seed"])
    results = {}
    for name in BRANCH:
        pipe = make_pipeline(cfg, name, y)
        oof = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
        results[name] = {"pr_auc": float(average_precision_score(y, oof))}
        if name == "baseline":
            # The one place accuracy is reported: it shows why it was abandoned.
            results[name]["accuracy"] = float(accuracy_score(y, oof >= 0.5))
        print(f"{name:20s} CV PR-AUC {results[name]['pr_auc']:.4f}", flush=True)
    return results


def tree_figure(cfg, X, y):
    """Printable depth-4 tree figure (fit on the training split only)."""
    pipe = make_pipeline(cfg, "decision_tree", y).fit(X, y)
    names = pipe["features"]["prep"].get_feature_names_out()
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(pipe["model"], feature_names=list(names), class_names=["repay", "default"],
              filled=True, impurity=False, proportion=True, fontsize=7, ax=ax)
    fig.savefig(artifact_path(cfg, "figures/decision_tree.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(skip_cv=False):
    cfg = load_config()
    X, y = load_train(cfg)
    if not skip_cv:
        update_metrics(cfg, "cv", cv_ladder(cfg, X, y))
    tree_figure(cfg, X, y)

    # The main model is LightGBM by design; the ladder shows how much it earns.
    final = make_pipeline(cfg, MAIN_MODEL, y).fit(X, y)
    joblib.dump(final, artifact_path(cfg, "pipeline.pkl"))
    print("saved artifacts/pipeline.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cv", action="store_true", help="only refit and save the main model")
    main(parser.parse_args().skip_cv)
