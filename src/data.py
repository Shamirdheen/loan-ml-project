"""Load the raw data and freeze the train/test split.

Owns:
    - Reading data/raw/application_train.csv (path from config.yaml -> data.raw_path).
    - One stratified train/test split on TARGET (test_size and seed from config.yaml).
    - Freezing the test set: saving the split indices to data/processed/ so every
      module and every teammate uses the identical split.

Does NOT own:
    - Any transform of feature values (imputing, scaling, encoding, engineered
      features). Those live in features.py, inside the Pipeline, so they are fit
      only on training folds.

Reads:  config.yaml, data/raw/application_train.csv (never written to)
Writes: data/processed/ (split indices)

What breaks if an assumption is wrong:
    - If the split isn't stratified, the ~8% default rate can drift between train
      and test, and every PR-AUC number downstream stops being comparable.
    - If anything calls .fit() here, preprocessing sees test rows and CV scores
      come out optimistically biased.
"""
