"""Group fairness metrics, protected-attribute ablation, and SHAP adverse-action reasons.

Primary fairness metric and tradeoff:
    EQUALIZED ODDS (set in config.yaml). Among applicants who would repay, every
    CODE_GENDER group should be approved at the same rate (equal true positive
    rate); among applicants who would default, every group should be approved at
    the same rate (equal false positive rate). The reported gap is the larger of
    the two between-group spreads.

    Why this one: default is a real outcome and base rates differ by group in
    this data. Equalized odds asks "given the same true repayment behaviour, is
    the model equally willing to lend?", which is the lending question. It also
    keeps the profit logic intact, because decisions still follow each person's
    risk.

    What it gives up: demographic parity (equal approval rates across groups).
    The two cannot both hold when default rates differ between groups, so a group
    with a higher default rate is approved less often even when the model treats
    its repayers and defaulters exactly like everyone else's. The parity gap is
    reported for context only and is not optimised. There is no per-group
    threshold adjustment, so equalized odds is measured here, not enforced.

Owns:
    - Group metrics on the calibrated pipeline's decisions, split by
      CODE_GENDER (check the actual values on load; don't assume M/F/XNA is
      the full set).
    - The proxy ablation: flip fairness.protected_in_model (drop CODE_GENDER if
      the main model uses it, add it if not), retrain, and report whether the
      gap moves. If dropping it doesn't close the gap, other features are acting
      as proxies.
    - SHAP-based adverse-action reasons for individual denials.

Does NOT own:
    - Model fitting, except the ablation retrain, which reuses train.py's code
      path instead of reimplementing it. The ablation is scored on training
      out-of-fold predictions; this module never loads the test set.
    - Calibration or threshold selection (evaluate.py).

Reads:  config.yaml -> fairness, artifacts/calibrated_pipeline.pkl,
        artifacts/metrics.json (the chosen threshold), data/processed/
Writes: artifacts/metrics.json (fairness section), artifacts/figures/

What breaks if an assumption is wrong:
    - If a rare CODE_GENDER value (e.g. XNA) is silently dropped or merged,
      the group metrics describe a population that doesn't match the data.
    - If metrics use a different threshold from evaluate.py, the fairness gap
      describes decisions the model never actually makes.
"""

import argparse

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from .data import artifact_path, load_config, load_raw, load_train, read_metrics, update_metrics
from .features import build_features
from .train import make_pipeline

MIN_GROUP = 100   # groups with fewer rows are listed but left out of the gaps


def group_table(group, y, p, threshold):
    """Per-group rates at the fixed profit threshold (approve iff p < threshold)."""
    df = pd.DataFrame({"group": np.asarray(group), "y": np.asarray(y), "approve": np.asarray(p) < threshold})
    rows = []
    for g, d in df.groupby("group", dropna=False):
        rows.append({
            "group": str(g), "n": len(d), "default_rate": d.y.mean(),
            "approval_rate": d.approve.mean(),
            # equalized-odds pair: approval among repayers, approval among defaulters
            "approve_given_repay": d.approve[d.y == 0].mean() if (d.y == 0).any() else np.nan,
            "approve_given_default": d.approve[d.y == 1].mean() if (d.y == 1).any() else np.nan,
        })
    return pd.DataFrame(rows).set_index("group")


def gaps(table):
    t = table[table.n >= MIN_GROUP]
    spread = lambda c: float(t[c].max() - t[c].min())
    return {"equalized_odds_gap": max(spread("approve_given_repay"), spread("approve_given_default")),
            "gap_approve_given_repay": spread("approve_given_repay"),
            "gap_approve_given_default": spread("approve_given_default"),
            "demographic_parity_gap": spread("approval_rate"),
            "groups_in_gap": list(t.index), "groups_excluded_small": list(table.index[table.n < MIN_GROUP])}


def proxy_check(cfg, X):
    """How well can the OTHER features predict CODE_GENDER? AUC 0.5 = no proxy at all.

    Hypothesis: occupation, organization and similar columns encode gender, so
    dropping the column alone will not close the gap. AUC well above 0.5 supports it.
    """
    prot = cfg["data"]["protected_attribute"]
    keep = X[prot].isin(["F", "M"])
    Xg, g = X[keep], (X.loc[keep, prot] == "M").astype(int)
    Xtr, Xva, gtr, gva = train_test_split(Xg, g, test_size=0.25, stratify=g, random_state=cfg["seed"])
    pipe = Pipeline([("features", build_features(cfg, "B", protected_in_model=False)),
                     ("model", LGBMClassifier(n_estimators=200, verbose=-1, random_state=cfg["seed"]))])
    pipe.fit(Xtr, gtr)
    return float(roc_auc_score(gva, pipe.predict_proba(Xva)[:, 1]))


def adverse_action_reasons(base_pipeline, X, top_k=3):
    """Top features pushing each applicant's risk UP, from SHAP on the uncalibrated model.

    Calibration is monotone, so it does not reorder what drives a score; explaining
    the base model explains the decision. Returns one list of strings per applicant.
    """
    feats = base_pipeline[:-1].transform(X)
    sv = shap.TreeExplainer(base_pipeline[-1]).shap_values(feats)
    sv = sv[1] if isinstance(sv, list) else sv
    sv = sv[..., 1] if sv.ndim == 3 else sv
    names = list(feats.columns)
    out = []
    for i in range(len(feats)):
        top = np.argsort(-sv[i])[:top_k]
        out.append([f"{names[j]} = {feats.iloc[i, j]}" for j in top if sv[i, j] > 0])
    return out


def group_figure(cfg, table, title, name):
    t = table[table.n >= MIN_GROUP]
    ax = t[["approval_rate", "approve_given_repay", "approve_given_default"]].plot.bar(figsize=(6, 4), rot=0)
    ax.set_title(title)
    ax.set_ylabel("rate")
    ax.figure.savefig(artifact_path(cfg, f"figures/{name}"), dpi=130, bbox_inches="tight")
    plt.close(ax.figure)


def main():
    cfg = load_config()
    if cfg["fairness"]["primary_metric"] != "equalized_odds":
        raise ValueError("fairness.primary_metric must be equalized_odds (the only metric implemented)")
    m = read_metrics(cfg)
    if "threshold" not in m or "test" not in m:
        raise RuntimeError("run evaluate.py first (threshold and test predictions are needed)")
    thr = m["threshold"]["value"]
    prot, id_col = cfg["data"]["protected_attribute"], cfg["data"]["id_column"]

    X, y = load_train(cfg)
    gender = X[prot]

    # Every comparison below uses calibrated OUT-OF-FOLD predictions on the training
    # split, so this module never loads the test set. The test set is touched once,
    # in evaluate.py; we only re-read the predictions it saved (a file read, not a new touch).
    oof = pd.read_csv(artifact_path(cfg, "oof_predictions.csv")).set_index(id_col)
    p_main = oof.loc[X[id_col], "p"].to_numpy()
    main_tbl = group_table(gender, y, p_main, thr)
    result = {"threshold": thr, "protected_in_model": cfg["fairness"]["protected_in_model"],
              "evaluated_on": "train out-of-fold, calibrated",
              "main": {"table": main_tbl.round(4).reset_index().to_dict("records"), **gaps(main_tbl)}}
    group_figure(cfg, main_tbl, "Main model: rates by CODE_GENDER (train OOF)", "fairness_main.png")

    # Same table from evaluate.py's saved test predictions, for the final report.
    tp = pd.read_csv(artifact_path(cfg, "test_predictions.csv"))
    raw_gender = load_raw(cfg).set_index(id_col)[prot]
    test_tbl = group_table(raw_gender.loc[tp[id_col]].to_numpy(), tp.y, tp.p, thr)
    result["main_test_reported"] = {"table": test_tbl.round(4).reset_index().to_dict("records"), **gaps(test_tbl)}

    # Ablation: flip whether the protected attribute is a model input, then compare
    # at the SAME threshold using the same folds and the same calibration as evaluate.py.
    if cfg["fairness"]["run_proxy_ablation"]:
        flipped = not cfg["fairness"]["protected_in_model"]
        folds = cfg["data"]["cv_folds"]
        skf = StratifiedKFold(folds, shuffle=True, random_state=cfg["seed"])
        cal = CalibratedClassifierCV(make_pipeline(cfg, "lightgbm", y, protected_in_model=flipped),
                                     method="isotonic", cv=folds)
        p_abl = cross_val_predict(cal, X, y, cv=skf, method="predict_proba")[:, 1]
        abl_tbl = group_table(gender, y, p_abl, thr)
        result["ablation"] = {"protected_in_model": flipped,
                              "table": abl_tbl.round(4).reset_index().to_dict("records"), **gaps(abl_tbl)}
        result["proxy_auc_gender_from_other_features"] = proxy_check(cfg, X)
        group_figure(cfg, abl_tbl, f"Ablation (protected_in_model={flipped}, train OOF)", "fairness_ablation.png")

    # Adverse-action demo on TRAINING applicants the calibrated out-of-fold model denies.
    base = joblib.load(artifact_path(cfg, "pipeline.pkl"))
    denied = X[oof.loc[X[id_col], "p"].to_numpy() >= thr].head(10)
    reasons = adverse_action_reasons(base, denied)
    pd.DataFrame({id_col: denied[id_col].to_numpy(), "reasons": ["; ".join(r) for r in reasons]}).to_csv(
        artifact_path(cfg, "adverse_action_examples.csv"), index=False)

    update_metrics(cfg, "fairness", result)
    for k in ("main", "ablation"):
        if k in result:
            print(k, {a: b for a, b in result[k].items() if a != "table"})
    print("proxy AUC:", result.get("proxy_auc_gender_from_other_features"))


if __name__ == "__main__":
    argparse.ArgumentParser(description="Group fairness report.").parse_args()
    main()
