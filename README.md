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

## Module order (one module at a time, reviewed before the next)

| Week | Module |
|---|---|
| 1 | `src/data.py` + `notebooks/01_eda.ipynb` |
| 1–2 | `src/features.py` |
| 2 | `src/model.py` + `src/train.py` |
| 3 | `src/evaluate.py` (needs `cost_assumptions` filled in `config.yaml`) |
| 3 | `src/fairness.py` |
| 4 | `app.py` (Streamlit demo; don't start it earlier) |
