"""Calibrate the model, compute metrics, sweep the profit threshold, and touch the test set once.

Owns:
    - Calibrating LightGBM with CalibratedClassifierCV (isotonic, split mode from
      config.yaml), fit on TRAINING data only. pipeline.pkl is the uncalibrated
      "before" model and the template CalibratedClassifierCV clones in "cv" mode.
      Reliability curves before and after.
    - Saving the calibrated model to artifacts/calibrated_pipeline.pkl. That file,
      not pipeline.pkl, is what fairness.py and app.py load.
    - PR-AUC (primary), ROC-AUC, and Brier score. Accuracy is reported once, in
      the baseline row, to show why it was abandoned at an 8% base rate.
    - The expected-profit threshold sweep, using loss_given_default and
      expected_interest (rates x AMT_CREDIT) from config.yaml. Must fail loudly
      if either is null; never fall back to a default.
    - Choosing the threshold on out-of-fold calibrated probabilities from the
      TRAINING split, and saving it to metrics.json for fairness.py and app.py.
    - The single, final evaluation on the held-out test set, which reports
      metrics and profit at the already-chosen threshold. It never picks it.

Does NOT own:
    - Fairness metrics (fairness.py).
    - Model fitting beyond calibration (train.py).

Reads:  config.yaml, artifacts/pipeline.pkl, data/processed/ (frozen split)
Writes: artifacts/calibrated_pipeline.pkl, artifacts/metrics.json, artifacts/figures/

What breaks if an assumption is wrong:
    - If calibration is fit on test rows, the "touch test exactly once" rule is
      broken and the reported numbers are no longer an honest estimate.
    - If the threshold is chosen on test rows, the reported profit is the best
      profit in hindsight, not an estimate of what the rule earns.
    - If uncalibrated probabilities reach the profit code, the chosen threshold
      optimizes against distorted risk estimates and the profit number is wrong.
"""

import argparse

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .data import artifact_path, load_config, load_test, load_train, read_metrics, update_metrics


def cost_rates(cfg):
    c = cfg["cost_assumptions"]
    if c["loss_given_default"] is None or c["expected_interest"] is None:
        raise ValueError("cost_assumptions in config.yaml are null; decide and document them first")
    return c["loss_given_default"], c["expected_interest"]


def profit(y, p, exposure, threshold, lgd, interest):
    """Total expected profit of approving every applicant with p < threshold.

    approve & repays -> +interest * exposure; approve & defaults -> -lgd * exposure.
    """
    approve = p < threshold
    gain = np.where(y == 0, interest * exposure, -lgd * exposure)
    return float(gain[approve].sum())


def sweep(y, p, exposure, lgd, interest, grid):
    return np.array([profit(y, p, exposure, t, lgd, interest) for t in grid])


def metrics_at(y, p):
    return {"pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "brier_score": float(brier_score_loss(y, p))}


def reliability_figure(cfg, y, p_raw, p_cal):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    for p, label in ((p_raw, "LightGBM, uncalibrated"), (p_cal, "LightGBM, isotonic")):
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=label)
    ax.set_xlabel("predicted default probability")
    ax.set_ylabel("observed default rate")
    ax.set_title("Reliability (training out-of-fold)")
    ax.legend()
    fig.savefig(artifact_path(cfg, "figures/reliability.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def profit_figure(cfg, grid, curve, best_t, theory_t):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, curve / 1e6)
    ax.axvline(best_t, color="C3", label=f"sweep optimum {best_t:.3f}")
    ax.axvline(theory_t, color="C2", ls="--", label=f"theory {theory_t:.3f}")
    ax.set_xlabel("approval threshold (approve if P(default) < t)")
    ax.set_ylabel("expected profit (millions)")
    ax.legend()
    fig.savefig(artifact_path(cfg, "figures/profit_curve.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(rerun_test=False):
    cfg = load_config()
    lgd, interest = cost_rates(cfg)
    if cfg["calibration"]["split"] != "cv" or cfg["calibration"]["method"] != "isotonic":
        raise NotImplementedError("only isotonic calibration with split: cv is implemented")
    if "test" in read_metrics(cfg) and not rerun_test:
        raise RuntimeError("the test set was already evaluated (metrics.json has a 'test' section); "
                           "re-running it breaks the touch-once rule. Pass --rerun-test only if the team agrees.")

    exposure_col = cfg["cost_assumptions"]["exposure_column"]
    id_col = cfg["data"]["id_column"]
    X, y = load_train(cfg)
    y = y.to_numpy()
    exposure = X[exposure_col].to_numpy()
    base = joblib.load(artifact_path(cfg, "pipeline.pkl"))
    folds = cfg["data"]["cv_folds"]
    skf = StratifiedKFold(folds, shuffle=True, random_state=cfg["seed"])
    make_cal = lambda: CalibratedClassifierCV(clone(base), method="isotonic", cv=folds)

    # Hypothesis: class weighting pushes raw probabilities well above the true
    # default rate; isotonic calibration pulls the reliability curve back onto
    # the diagonal and lowers Brier. The profit rule needs true probabilities.
    p_raw = cross_val_predict(clone(base), X, y, cv=skf, method="predict_proba")[:, 1]
    p_cal = cross_val_predict(make_cal(), X, y, cv=skf, method="predict_proba")[:, 1]
    reliability_figure(cfg, y, p_raw, p_cal)

    # Hypothesis: with calibrated p, approving iff p*LGD < (1-p)*interest is optimal,
    # i.e. threshold = interest / (interest + LGD), the same for every loan size
    # (both sides scale with exposure). The sweep checks that on real data.
    grid = np.linspace(0.01, 0.99, 197)
    curve = sweep(y, p_cal, exposure, lgd, interest, grid)
    best_t = float(grid[curve.argmax()])
    theory_t = interest / (interest + lgd)
    profit_figure(cfg, grid, curve, best_t, theory_t)
    sensitivity = []
    for l in (0.4, 0.5, 0.6, 0.7, 0.8):
        for i in (0.10, 0.15, 0.20):
            t = i / (i + l)
            sensitivity.append({"lgd": l, "interest": i, "threshold": t,
                                "train_oof_profit": profit(y, p_cal, exposure, t, l, i)})

    cal = make_cal().fit(X, y)
    joblib.dump(cal, artifact_path(cfg, "calibrated_pipeline.pkl"))
    pd.DataFrame({id_col: X[id_col].to_numpy(), "y": y, "p": p_cal}).to_csv(
        artifact_path(cfg, "oof_predictions.csv"), index=False)

    update_metrics(cfg, "train_oof", {
        "uncalibrated": metrics_at(y, p_raw), "calibrated": metrics_at(y, p_cal),
        "profit_at_chosen": profit(y, p_cal, exposure, best_t, lgd, interest),
        "profit_approve_all": profit(y, p_cal, exposure, 1.01, lgd, interest)})
    update_metrics(cfg, "threshold", {"value": best_t, "theory": theory_t, "lgd": lgd,
                                      "interest": interest, "chosen_on": "train_oof_calibrated"})
    update_metrics(cfg, "cost_sensitivity", sensitivity)

    # ---- the one and only test-set evaluation: threshold is already fixed ----
    X_te, y_te = load_test(cfg)
    y_te = y_te.to_numpy()
    exp_te = X_te[exposure_col].to_numpy()
    p_te = cal.predict_proba(X_te)[:, 1]
    test = {"calibrated": metrics_at(y_te, p_te),
            "uncalibrated": metrics_at(y_te, base.predict_proba(X_te)[:, 1]),
            "threshold": best_t,
            "profit_at_threshold": profit(y_te, p_te, exp_te, best_t, lgd, interest),
            "profit_approve_all": profit(y_te, p_te, exp_te, 1.01, lgd, interest),
            "approval_rate": float((p_te < best_t).mean()),
            "n": int(len(y_te))}
    update_metrics(cfg, "test", test)
    pd.DataFrame({id_col: X_te[id_col].to_numpy(), "y": y_te, "p": p_te}).to_csv(
        artifact_path(cfg, "test_predictions.csv"), index=False)
    print(read_metrics(cfg)["test"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-test", action="store_true")
    main(parser.parse_args().rerun_test)
