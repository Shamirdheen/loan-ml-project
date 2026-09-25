"""Define the 5-model ladder once, from config.yaml. Estimator objects only.

Owns:
    - Exactly five unfitted estimators: majority-class baseline, logistic
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
