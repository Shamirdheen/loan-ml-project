# Loan Approval Risk Model

Home Credit default-risk classifier. Project rules are in `CLAUDE.md`; the
technical spec is in `ARCHITECTURE_LOAN.md`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

## Data

Download `application_train.csv` from Kaggle's **Home Credit Default Risk**
competition and put it in `data/raw/`. The file is gitignored; never commit it
and never write to `data/raw/`.

Then create the frozen train/test split **once**, as a team, and commit it:

```bash
python -m src.data          # writes data/processed/split.csv, refuses to overwrite
```

Everyone else just pulls `split.csv`. Never re-run with `--force` unless the
whole team agrees to redo every result.

## Module order (one module at a time, reviewed before the next)

| Week | Module |
|---|---|
| 1 | `src/data.py` + `notebooks/01_eda.ipynb` |
| 1–2 | `src/features.py` |
| 2 | `src/model.py` + `src/train.py` |
| 3 | `src/evaluate.py` (needs `cost_assumptions` filled in `config.yaml`) |
| 3 | `src/fairness.py` |
| 4 | `app.py` (Streamlit demo; don't start it earlier) |

## Run order (once the CSV is in `data/raw/`)

```bash
python -m src.data        # frozen split (once)
python -m src.train       # 5-model CV ladder + depth-4 tree figure + artifacts/pipeline.pkl
python -m src.evaluate    # calibration, profit threshold, the ONE test evaluation
python -m src.fairness    # equalized-odds report, CODE_GENDER ablation, SHAP reasons
streamlit run app.py      # demo
```

`evaluate.py` refuses to run twice (the test set is touched once). All modules were
smoke-tested on a synthetic dataset; no result on the real data exists yet.
