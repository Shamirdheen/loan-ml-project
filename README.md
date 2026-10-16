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
streamlit run app.py      # faculty demo UI
```

`evaluate.py` refuses to run twice (the test set is touched once).

## Faculty demo (`streamlit run app.py`)

The app loads `artifacts/calibrated_pipeline.pkl` and **never** the test set. Walk the tabs in order:

1. **Decide** — three sample applicants (Typical / Strong file / High risk), live APPROVE/DENY, expected profit, denial reasons, loan-size what-if.
2. **Model** — held-out numbers from `metrics.json` (one test pass), 5-model PR-AUC ladder, reliability curve.
3. **Profit lab** — move LGD / interest / fixed cost on **training** out-of-fold predictions. Sliders do not change the deployed threshold.
4. **Fairness** — equalized odds by `CODE_GENDER`, plus the gender-column ablation.

Needs a finished local run (`train` → `evaluate` → `fairness`) so `calibrated_pipeline.pkl`, `metrics.json`, and `oof_predictions.csv` exist.

## Web demo (HTML/CSS/JS, `server.py` + `web/`)

Same Decide tab as the Streamlit app, as a plain web page with amounts in ₹.
`server.py` (FastAPI) loads the same `calibrated_pipeline.pkl` and serves `web/`.
Model / Profit lab / Fairness tabs are placeholders for now.

```powershell
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000   (use 127.0.0.1, not localhost)
```

No trained model on this machine? Start it with `$env:LOAN_DEMO="1"` first. Scores then
come from a hand-written formula in `demo_mock.py` (yellow "Demo mode" pill + banner) —
for learning the screens only, never for reported numbers. A captioned walkthrough of
demo mode is in `demo_video/loan_risk_studio_demo.webm`.

**Status:** the full pipeline has run on the real data. Results, and what's still
left to do, are in [`PROGRESS.md`](PROGRESS.md).
