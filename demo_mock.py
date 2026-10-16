"""Stand-in scorer so the web UI can be walked through on a PC without the trained model.

NOT THE MODEL. A hand-written logistic formula over the form fields, tuned only
so the three presets land roughly where the real model puts them (Strong file
well below the threshold, Typical near the ~8% base rate, High risk above it).
Nothing here is fitted to data, and its numbers must never appear in the report.

Used only when server.py is started with LOAN_DEMO=1 AND the real artifacts are
missing. The page shows a yellow "demo mode" banner whenever this is active.

What breaks if an assumption is wrong:
    - If someone demos with LOAN_DEMO=1 set and the banner is ignored, faculty
      see made-up probabilities. Unset LOAN_DEMO before the real demo.
"""
import numpy as np

# Rough Home Credit training medians, typed in by hand (the CSV is not on this PC).
TYPICAL = {
    "income": 147_150, "credit": 513_531, "annuity": 24_903, "age": 43,
    "no_job": False, "employed": 6, "education": "Secondary / secondary special",
    "ext1": 0.51, "ext2": 0.57, "ext3": 0.54,
}
EDU_SHIFT = {
    "Academic degree": -0.5, "Higher education": -0.3, "Incomplete higher": 0.0,
    "Secondary / secondary special": 0.0, "Lower secondary": 0.4,
}


def _contributions(a, credit):
    """Each term's push on the log-odds; positive = riskier. Mirrors how denial reasons read."""
    ext = np.mean([a.ext1, a.ext2, a.ext3])
    return {
        "EXT_SOURCE (mean)": (-5.0 * (ext - 0.55), f"{ext:.2f}"),
        "CREDIT_TO_INCOME": (0.08 * (credit / a.income - 3.5), f"{credit / a.income:.2f}"),
        "ANNUITY_TO_INCOME": (1.5 * (a.annuity * credit / a.credit / a.income - 0.17),
                              f"{a.annuity * credit / a.credit / a.income:.3f}"),
        "AGE_YEARS": (-0.15 * (a.age - 43) / 10, f"{a.age}"),
        "YEARS_EMPLOYED": (0.25 if a.no_job else -0.2 * (a.employed - 6) / 5,
                           "not employed" if a.no_job else f"{a.employed}"),
        "NAME_EDUCATION_TYPE": (EDU_SHIFT.get(a.education, 0.0), a.education),
    }


def predict(a, loans):
    loans = np.atleast_1d(np.asarray(loans, dtype=float))
    out = []
    for credit in loans:
        logit = -2.44 + sum(v for v, _ in _contributions(a, credit).values())
        out.append(1 / (1 + np.exp(-logit)))
    return np.array(out)


def reasons(a, top_k=3):
    parts = sorted(_contributions(a, a.credit).items(), key=lambda kv: -kv[1][0])
    return [{"feature": name, "value": shown} for name, (v, shown) in parts[:top_k] if v > 0]
