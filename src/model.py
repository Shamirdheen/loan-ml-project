"""Define the 5-model ladder once, from config.yaml. Estimator objects only.

Owns:
    - Exactly five unfitted estimators, built by importing each `type` path in
      config.yaml and passing `params`: majority-class baseline
      (DummyClassifier, strategy=prior), logistic
      regression, decision tree (depth 4, for the printable figure), random
      forest, and LightGBM.
    - Class-imbalance handling through class_weight / scale_pos_weight. SMOTE
      is not the default. If it is ever added, it's an explicit ablation.
    - scale_pos_weight should be computed from the training split's actual
      class ratio, not taken from the placeholder value in config.yaml.

Does NOT own:
    - Preprocessing (features.py), fitting (train.py), or evaluation (evaluate.py).
    - Any model beyond these five. No XGBoost/CatBoost, and no neural net as the
      main model.

Reads:  config.yaml -> models
Writes: nothing

What breaks if an assumption is wrong:
    - If the class ratio shifts and scale_pos_weight stays hardcoded, LightGBM's
      raw probabilities get distorted differently. Calibration in evaluate.py
      compensates, but only if it's actually applied.
"""

import importlib

# Which preprocessing branch (features.py) each model needs.
BRANCH = {"baseline": "A", "logistic_regression": "A", "decision_tree": "A",
          "random_forest": "A", "lightgbm": "B"}


def _import(path):
    module, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def build_models(cfg, y_train):
    """Return {name: unfitted estimator} for the five models in config.yaml."""
    models = {}
    for spec in cfg["models"]:
        cls, params = _import(spec["type"]), dict(spec["params"])
        if spec["name"] == "lightgbm":
            # Hypothesis: weighting defaults by the real negative/positive ratio
            # lifts recall on the rare class; calibration later repairs the
            # probability distortion this weighting causes.
            pos = int(y_train.sum())
            params["scale_pos_weight"] = (len(y_train) - pos) / pos
            params["verbose"] = -1
        if "random_state" in cls().get_params():
            params.setdefault("random_state", cfg["seed"])
        models[spec["name"]] = cls(**params)
    if set(models) != set(BRANCH):
        raise ValueError(f"config.yaml models must be exactly {sorted(BRANCH)}, got {sorted(models)}")
    return models
