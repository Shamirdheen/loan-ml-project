# Progress — what's done and what's left

Last updated: 2026-10-15. All numbers below come from `artifacts/metrics.json`
after the first full run on the real data (`metrics.json` itself is gitignored;
regenerate it with the run order in `README.md`).

## Done

### Pipeline runs end to end on the real data
`application_train.csv`: 307,511 rows × 122 columns, 8.07% default rate.
`CODE_GENDER`: F 202,448 · M 105,059 · XNA 4 (too small for gap metrics; listed, excluded).

| Step | Result |
|---|---|
| `src.data` | Frozen stratified split committed: 246,008 train / 61,503 test, both 8.07% default |
| `src.train` | 5-model ladder, 5-fold CV PR-AUC on the training split (below) |
| `src.evaluate` | Isotonic calibration, threshold chosen on train out-of-fold, test set evaluated **once** |
| `src.fairness` | Equalized-odds report + `CODE_GENDER` ablation, on train out-of-fold only |

**Model ladder (CV PR-AUC, training split)**

| Model | PR-AUC |
|---|---|
| baseline (prior) | 0.0807 — equals the base rate; its accuracy is 91.9%, which is why accuracy was dropped |
| decision tree (depth 4) | 0.1855 |
| random forest | 0.2206 |
| logistic regression | 0.2223 |
| **LightGBM** | **0.2314** |

LightGBM wins, but only narrowly over logistic regression — worth saying in the report.

**Calibration** — Brier score, uncalibrated → isotonic: 0.1555 → 0.0681 (train OOF),
0.1603 → 0.0676 (test). Calibration also nudged PR-AUC up (0.231 → 0.239 OOF).

**Profit threshold** — chosen on train OOF: **0.19** (theory with no fixed cost: 0.20).

| Held-out test set (touched once) | Value |
|---|---|
| PR-AUC / ROC-AUC (calibrated) | 0.249 / 0.764 |
| Approval rate | 91.2% |
| Profit at threshold | 3,625,074,021 |
| Profit approving everyone | 3,433,243,735 |
| Gain from the model | +191,830,286 (+5.6%) |

**Fairness** (equalized odds primary; train OOF, threshold 0.19)

| | Main model (gender NOT an input) | Ablation (gender added) |
|---|---|---|
| Equalized-odds gap | 7.4% | 9.9% |
| Demographic-parity gap | 4.9% | 6.0% |

- Adding `CODE_GENDER` as an input *widens* the gap, so keeping it out (`protected_in_model: false`) is supported.
- Other features predict gender with AUC **0.904**: dropping the column doesn't remove its influence (proxies).
- Test-set equalized-odds gap (report only): 10.4%. The gap is driven by approval among
  *defaulters* (F 73.7% vs M 66.2%), not among repayers (94.8% vs 90.9%).

### Code changes this session
- **Fixed cost per loan** (`evaluate.py`, `config.yaml`): profit =
  `AMT_CREDIT·((1−p)·interest − p·LGD) − fixed_cost`; fails loudly if null; reports the
  zero-risk break-even loan size (`fixed_cost / interest`) and plots threshold vs loan size.
- **Streamlit app rebuilt** (`app.py`) — 4 interactive tabs: Decide (gauge, approve/deny card,
  expected profit, denial reasons, what-if over loan size), Model (test metrics, PR-AUC ladder,
  reliability, tree), Profit lab (cost sliders → live profit curve and best threshold, train OOF
  only), Fairness (threshold slider → group rates and gaps, ablation). Fixed a startup crash
  (it imported a `CATEGORICAL` list that no longer exists). Run: `streamlit run app.py`.
- Verified by reading the code: fairness ablation never touches the test set; categorical
  columns are detected by dtype, not a hardcoded list.

## Left to do

### Decisions the team owes (in `config.yaml`)
- [ ] **`fixed_cost_per_loan`** — currently a provisional `0`, chosen for time. Set a real
      value and write the reasoning. Any value under 6,750 changes nothing (smallest loan is 45,000).
- [ ] **Per-loan vs single global threshold** — with a fixed cost the per-loan rule is optimal;
      switching touches `fairness.py` and `app.py`.

### Features not started
- [ ] **1. `src/ablation.py`** — drop each engineered feature, retrain LightGBM (5-fold CV,
      train only), PR-AUC delta table `artifacts/ablation.csv` + bar chart; one-line hypothesis per feature.
- [ ] **2. approve / human_review / deny** — `review_band: null` in config (team sets it);
      evaluate fails while null; report approval rate, review rate, profit with/without the band.
      Needs a cost per human review too.
- [ ] **3. `scripts/make_model_card.py`** — `artifacts/MODEL_CARD.md` from `metrics.json`,
      TODO where a value is missing. Do last.
- [ ] **5. Profit by model** — each of the 5 models (calibrated) gets its own threshold on train
      OOF; test profit / approval rate / PR-AUC table `artifacts/model_profit_comparison.csv` + chart.
      Must run inside evaluate's single test pass, or it breaks the touch-once rule.
- [ ] **6. Threshold stability** — 200 bootstrap resamples of train OOF; median threshold,
      95% interval, profit interval, one chart. No test set.

### Housekeeping
- [ ] `notebooks/01_eda.ipynb` — flag the `DAYS_EMPLOYED = 365243` placeholder (55,374 rows, 18%)
      and eyeball the 16 dtype-detected categorical columns.
- [ ] Re-running evaluate after any change above needs `--rerun-test` and team agreement
      (the test set has now been used once).
