"""Load the raw data and freeze the train/test split.

Owns:
    - Reading data/raw/application_train.csv (path from config.yaml -> data.raw_path).
    - One stratified train/test split on TARGET (test_size and seed from config.yaml).
    - Freezing the test set: saving split membership by SK_ID_CURR to
      data/processed/split.csv so every module and every teammate uses the
      identical split. That file is committed to git.

Does NOT own:
    - Any transform of feature values (imputing, scaling, encoding, engineered
      features). Those live in features.py, inside the Pipeline, so they are fit
      only on training folds.

Reads:  config.yaml, data/raw/application_train.csv (never written to)
Writes: data/processed/split.csv

Usage:
    python -m src.data            # create the split (refuses if one exists)
    python -m src.data --force    # overwrite it; invalidates every past result

What breaks if an assumption is wrong:
    - If the split isn't stratified, the ~8% default rate can drift between train
      and test, and every PR-AUC number downstream stops being comparable.
    - If anything calls .fit() here, preprocessing sees test rows and CV scores
      come out optimistically biased.
    - If SK_ID_CURR is not unique, membership by ID is ambiguous. load_raw checks
      this and fails rather than guessing.
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    # LOAN_CONFIG lets tests point every module at a scratch config (synthetic
    # data, scratch artifacts dir) without touching the real paths.
    path = path or os.environ.get("LOAN_CONFIG") or ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def artifact_path(cfg, name):
    """Path inside the artifacts dir (created on demand)."""
    base = ROOT / cfg.get("artifacts_dir", "artifacts")
    (base / "figures").mkdir(parents=True, exist_ok=True)
    return base / name


def read_metrics(cfg):
    p = artifact_path(cfg, "metrics.json")
    return json.loads(p.read_text()) if p.exists() else {}


def update_metrics(cfg, section, values):
    """Merge one section into metrics.json so each module only writes its own part."""
    m = read_metrics(cfg)
    m[section] = values
    artifact_path(cfg, "metrics.json").write_text(json.dumps(m, indent=2))


def load_raw(cfg):
    """Read the raw CSV and check the few facts every later module relies on."""
    d = cfg["data"]
    df = pd.read_csv(ROOT / d["raw_path"])

    id_col, target = d["id_column"], d["target_column"]
    for col in (id_col, target, d["protected_attribute"]):
        if col not in df.columns:
            raise KeyError(f"Expected column {col!r} not found in {d['raw_path']}")
    if df[id_col].duplicated().any():
        raise ValueError(f"{id_col} is not unique; split-by-ID would be ambiguous")
    if set(df[target].unique()) != {0, 1}:
        raise ValueError(f"{target} must be binary 0/1, got {sorted(df[target].unique())}")
    return df


def make_split(cfg, force=False):
    """Create and save the one train/test split. Refuses to overwrite by default."""
    d = cfg["data"]
    split_path = ROOT / d["split_path"]
    if split_path.exists() and not force:
        raise FileExistsError(
            f"{d['split_path']} already exists. The test set is frozen; pass "
            "force=True (--force) only if the whole team agrees to redo every result."
        )

    df = load_raw(cfg)
    # Why stratify: at an ~8% positive rate, a random 20% test slice can land
    # noticeably above or below the train rate by chance. Stratifying pins
    # both to the same rate so PR-AUC is comparable between CV and test.
    train_ids, test_ids = train_test_split(
        df[d["id_column"]],
        test_size=d["test_size"],
        stratify=df[d["target_column"]],
        random_state=cfg["seed"],
    )
    split = pd.concat([
        pd.DataFrame({d["id_column"]: train_ids, "split": "train"}),
        pd.DataFrame({d["id_column"]: test_ids, "split": "test"}),
    ]).sort_values(d["id_column"])

    split_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(split_path, index=False)
    return df, split


def _load_part(cfg, part):
    d = cfg["data"]
    split_path = ROOT / d["split_path"]
    if not split_path.exists():
        raise FileNotFoundError(f"{d['split_path']} missing; run `python -m src.data` first")

    df = load_raw(cfg)
    split = pd.read_csv(split_path)
    if set(split[d["id_column"]]) != set(df[d["id_column"]]):
        raise ValueError("split.csv IDs don't match the raw CSV; was the data file replaced?")

    ids = split.loc[split["split"] == part, d["id_column"]]
    rows = df[df[d["id_column"]].isin(ids)]
    X = rows.drop(columns=[d["target_column"]])
    y = rows[d["target_column"]]
    return X, y


def load_train(cfg):
    """Training split as (X, y). X still contains the ID column; features.py drops it."""
    return _load_part(cfg, "train")


def load_test(cfg):
    """Held-out test split as (X, y).

    Only evaluate.py calls this, exactly once, for the final report. Calling it
    anywhere else (EDA, tuning, threshold choice) breaks the honesty of every
    reported number.
    """
    return _load_part(cfg, "test")


def _summary(df, split, cfg):
    d = cfg["data"]
    merged = df[[d["id_column"], d["target_column"]]].merge(split, on=d["id_column"])
    print(f"rows: {len(df):,}   columns: {df.shape[1]}")
    print(merged.groupby("split")[d["target_column"]].agg(rows="size", positive_rate="mean"))
    # CLAUDE.md: verify the protected attribute's actual values, don't assume.
    print(f"\n{d['protected_attribute']} values (including NaN):")
    print(df[d["protected_attribute"]].value_counts(dropna=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the frozen train/test split.")
    parser.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = parser.parse_args()

    cfg = load_config()
    df, split = make_split(cfg, force=args.force)
    print(f"Saved {cfg['data']['split_path']}\n")
    _summary(df, split, cfg)
