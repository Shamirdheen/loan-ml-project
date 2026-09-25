"""Define the preprocessing Pipelines and engineered features. Nothing is fit here.

Owns:
    - The single source of truth for column-type lists: categorical, numeric,
      and ID columns to drop. Both branches read from these lists, so a schema
      change updates both at once.
    - Branch A (logistic regression, random forest): a ColumnTransformer that
      one-hot encodes categoricals, median-imputes numerics, and standard-scales them.
    - Branch B (LightGBM): a ColumnTransformer that uses native categorical
      handling and passes raw nulls through. LightGBM learns a default direction
      for missing values, and imputing first would throw that away.
    - Engineered features, each with a written hypothesis stated before it.

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
