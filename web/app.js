// Loan Risk Studio frontend. All scoring happens in server.py; this file only
// collects the form, calls /api/score, and draws the result. If a number on the
// page looks wrong, check the API response first - nothing is computed here
// except formatting and two display-only ratios.

const COLORS = {
  blue: "#1b4f8a", orange: "#c45c26", muted: "#7a756c", axis: "#d4cfc4",
  good: "#0f7a3a", bad: "#c0392b",
};
const FIELDS = ["income", "credit", "annuity", "age", "no_job", "employed",
                "education", "ext1", "ext2", "ext3"];
const PRESET_NOTES = { "Typical": "median applicant", "Strong file": "good scores", "High risk": "big loan, low scores" };
// Plain-English names for the features the model most often cites. Unknown names show as-is.
const FEATURE_LABELS = {
  EXT_SOURCE_1: "Bureau score 1", EXT_SOURCE_2: "Bureau score 2", EXT_SOURCE_3: "Bureau score 3",
  "EXT_SOURCE (mean)": "Average bureau score", AMT_CREDIT: "Loan amount", AMT_ANNUITY: "Yearly repayment",
  AMT_GOODS_PRICE: "Price of goods", AMT_INCOME_TOTAL: "Annual income", DAYS_BIRTH: "Age",
  DAYS_EMPLOYED: "Time employed", NAME_EDUCATION_TYPE: "Education", CREDIT_TO_INCOME: "Loan ÷ income",
  ANNUITY_TO_INCOME: "Repayment ÷ income", CODE_GENDER: "Gender", AGE_YEARS: "Age",
  YEARS_EMPLOYED: "Years employed",
};

const $ = (sel) => document.querySelector(sel);
const form = $("#applicant");
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

// ------------------------------------------------------------------ rupees
const inr = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
const rupees = (v) => inr.format(Math.round(v));          // ₹5,13,531 (Indian digit grouping)
function rupeesShort(v) {                                    // ₹5.1 L, ₹1.2 Cr, ₹45K
  const sign = v < 0 ? "−" : "", a = Math.abs(v);
  if (a >= 1e7) return `${sign}₹${+(a / 1e7).toFixed(2)} Cr`;
  if (a >= 1e5) return `${sign}₹${+(a / 1e5).toFixed(2)} L`;
  if (a >= 1e3) return `${sign}₹${+(a / 1e3).toFixed(1)}K`;
  return `${sign}₹${Math.round(a)}`;
}
function rupeesWords(v) {
  if (!v) return "";
  if (v >= 1e7) return `${+(v / 1e7).toFixed(2)} crore`;
  if (v >= 1e5) return `${+(v / 1e5).toFixed(2)} lakh`;
  return `${+(v / 1e3).toFixed(1)} thousand`;
}

let meta = null;
let pending = null;    // AbortController for the in-flight request
let debounce = null;

// ------------------------------------------------------------------ tabs
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".panel").forEach((p) => {
      p.hidden = p.id !== `tab-${btn.dataset.tab}`;
    });
    // Plotly sizes charts to their container; hidden containers have no size.
    window.dispatchEvent(new Event("resize"));
  });
});

// ------------------------------------------------------------------ form
function readForm() {
  const f = form.elements;
  return {
    income: Number(f.income.value), credit: Number(f.credit.value),
    annuity: Number(f.annuity.value), age: Number(f.age.value),
    no_job: f.no_job.checked, employed: Number(f.employed.value),
    education: f.education.value,
    ext1: Number(f.ext1.value), ext2: Number(f.ext2.value), ext3: Number(f.ext3.value),
  };
}

function writeForm(values) {
  const f = form.elements;
  for (const k of FIELDS) {
    if (f[k].type === "checkbox") f[k].checked = values[k];
    else f[k].value = values[k];
  }
  syncOutputs();
}

function syncOutputs() {
  const f = form.elements;
  f.employed.disabled = f.no_job.checked;
  document.querySelectorAll("output[data-for]").forEach((o) => {
    const v = Number(f[o.dataset.for].value);
    o.textContent = (o.dataset.for.startsWith("ext") ? v.toFixed(2) : v) + (o.dataset.unit || "");
  });
  document.querySelectorAll("[data-words]").forEach((el) => {
    el.textContent = rupeesWords(Number(f[el.dataset.words].value));
  });
  const a = readForm();
  $("#ratio-lti").textContent = a.income ? `${(a.credit / a.income).toFixed(1)}×` : "—";
  $("#ratio-burden").textContent = a.income ? pct(a.annuity / a.income, 0) : "—";
}

form.addEventListener("input", () => {
  syncOutputs();
  document.querySelectorAll(".presets button").forEach((b) => b.classList.remove("active"));
  clearTimeout(debounce);
  debounce = setTimeout(score, 250);   // sliders fire many events; score once they settle
});
form.addEventListener("submit", (e) => e.preventDefault());

// ------------------------------------------------------------------ scoring
// Range checks only. checkValidity() would also reject values off the spinner
// step (e.g. a training-median income of 147,150), which are perfectly valid.
function inRange() {
  return [...form.querySelectorAll("input[type=number]")].every(
    (el) => !el.validity.valueMissing && !el.validity.rangeUnderflow && !el.validity.rangeOverflow);
}

async function score() {
  if (!inRange()) return;   // the API would reject out-of-range values anyway
  pending?.abort();
  pending = new AbortController();
  setBusy(true);
  try {
    const res = await fetch("/api/score", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readForm()), signal: pending.signal,
    });
    const body = await res.json();
    if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
    $("#load-error").hidden = true;
    render(readForm(), body);
  } catch (err) {
    if (err.name !== "AbortError") showError(`Scoring failed: ${err.message}`);
  } finally {
    setBusy(false);
  }
}

function setBusy(on) {
  $(".result").classList.toggle("busy", on);
  document.querySelectorAll(".chart").forEach((c) => c.classList.toggle("busy", on));
}

function riskBand(p, t) {
  if (p < t / 2) return "Low risk";
  if (p < t) return "Moderate risk";
  if (p < t * 2) return "High risk";
  return "Very high risk";
}

function render(a, r) {
  const v = $("#verdict");
  v.className = `verdict ${r.approved ? "ok" : "no"}`;
  v.querySelector(".kicker").textContent = r.demo ? "Live decision · demo formula" : "Live decision · calibrated LightGBM";
  v.querySelector(".headline").textContent = r.approved ? "✓ APPROVE" : "✕ DENY";
  v.querySelector(".sub").textContent =
    `${pct(r.p)} risk is ${r.approved ? "below" : "at or above"} the ${pct(r.threshold)} threshold.`;
  $("#risk-big").textContent = pct(r.p);
  $("#risk-band").textContent = riskBand(r.p, r.threshold);

  // Risk bar: scale grows if the applicant is off the default 0-40% range.
  const max = Math.max(0.4, Math.ceil((r.p + 0.05) * 10) / 10);
  $("#riskbar-safe").style.width = `${(r.threshold / max) * 100}%`;
  $("#riskbar-marker").style.left = `${Math.min(r.p / max, 1) * 100}%`;
  const thr = $("#riskbar-thr");
  thr.style.left = `${(r.threshold / max) * 100}%`;
  thr.textContent = `threshold ${pct(r.threshold, 0)}`;
  $("#riskbar-max").textContent = pct(max, 0);

  $("#expected").textContent = rupees(r.expected_profit);
  const tag = $("#expected-tag");
  tag.textContent = r.expected_profit > 0 ? "▲ profitable" : "▼ loss-making";
  tag.className = r.expected_profit > 0 ? "up" : "down";
  $("#breakeven").textContent = pct(r.breakeven);

  $("#formula").textContent =
    `E[profit] = ${rupees(a.credit)} × ((1 − ${r.p.toFixed(3)})×${meta.interest.toFixed(2)} − ` +
    `${r.p.toFixed(3)}×${meta.lgd.toFixed(2)}) − ${rupees(meta.fixed)} = ${rupees(r.expected_profit)}`;
  $("#formula-words").textContent =
    `If repaid (${pct(1 - r.p)} chance) we earn ${pct(meta.interest, 0)} of the loan; if it defaults ` +
    `(${pct(r.p)} chance) we lose ${pct(meta.lgd, 0)} of it. Approve only when risk is below ${pct(r.threshold)}.`;

  $("#reasons").hidden = r.approved;
  const list = $("#reason-list");
  list.replaceChildren();
  if (!r.approved) {
    if (!r.reasons.length) {
      const li = document.createElement("li");
      li.textContent = "No single feature dominates this denial.";
      list.append(li);
    }
    for (const { feature, value } of r.reasons) {
      const li = document.createElement("li");
      const left = document.createElement("div");
      const name = document.createElement("b");
      name.textContent = FEATURE_LABELS[feature] || feature;
      const raw = document.createElement("code");
      raw.textContent = feature;
      left.append(name, raw);
      const val = document.createElement("span");
      val.textContent = value;
      li.append(left, val);
      list.append(li);
    }
  }

  drawWhatIf(a, r);
}

// ------------------------------------------------------------------ charts
const BASE_LAYOUT = {
  margin: { l: 70, r: 16, t: 48, b: 52 },
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "IBM Plex Sans, sans-serif", color: "#1c1917" },
  hovermode: "closest", showlegend: false, height: 320,
};
const PLOT_CONFIG = { displayModeBar: false, responsive: true };
const axis = (extra) => ({ showgrid: false, linecolor: COLORS.axis, ...extra });
const yaxis = (extra) => ({ gridcolor: "rgba(122,117,108,0.18)", zeroline: false, ...extra });

// Log-scale loan axis labelled in lakh, since Plotly's own formats only know western grouping.
const LOAN_TICKS = [50e3, 1e5, 2e5, 5e5, 10e5, 20e5, 40e5];
const loanAxis = () => axis({
  type: "log", title: { text: "loan amount" },
  tickvals: LOAN_TICKS, ticktext: LOAN_TICKS.map(rupeesShort),
});

function niceRupeeTicks(values, n = 5) {
  const lo = Math.min(...values, 0), hi = Math.max(...values, 0);
  const raw = (hi - lo) / n || 1;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const ticks = [];
  for (let t = Math.floor(lo / step) * step; t <= hi + step / 2; t += step) ticks.push(t);
  return { tickvals: ticks, ticktext: ticks.map(rupeesShort) };
}

function drawWhatIf(a, r) {
  const { loans, p, profit } = r.whatif;
  const loanText = loans.map(rupees);
  const current = (y, text) => ({
    x: [a.credit], y: [y], mode: "markers", customdata: [[rupees(a.credit), text]],
    hovertemplate: "Current loan %{customdata[0]}<br>%{customdata[1]}<extra></extra>",
    marker: { size: 13, color: COLORS.orange, line: { width: 2, color: "white" } },
  });

  Plotly.react("whatif-risk", [{
    x: loans, y: p, mode: "lines", line: { color: COLORS.blue, width: 2.5, shape: "spline" },
    customdata: loanText, hovertemplate: "Loan %{customdata}<br>Risk %{y:.1%}<extra></extra>",
  }, current(r.p, pct(r.p))], {
    ...BASE_LAYOUT, title: { text: "Default risk vs loan size", font: { size: 15 } },
    xaxis: loanAxis(),
    yaxis: yaxis({ tickformat: ".0%", title: { text: "P(default)" }, rangemode: "tozero" }),
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: r.threshold, y1: r.threshold,
               line: { color: COLORS.muted, dash: "dash" } }],
    annotations: [{ xref: "paper", x: 0.01, y: r.threshold, text: `approve below ${pct(r.threshold, 0)}`,
                    showarrow: false, xanchor: "left", yanchor: "bottom",
                    font: { color: COLORS.muted, size: 12 } }],
  }, PLOT_CONFIG);

  const colors = profit.map((v) => (v >= 0 ? COLORS.good : COLORS.bad));
  Plotly.react("whatif-profit", [{
    x: loans, y: profit, mode: "lines+markers", line: { color: COLORS.blue, width: 2.5, shape: "spline" },
    marker: { size: 5, color: colors },
    customdata: loans.map((l, i) => [rupees(l), rupees(profit[i])]),
    hovertemplate: "Loan %{customdata[0]}<br>Expected profit %{customdata[1]}<extra></extra>",
  }, current(r.expected_profit, `Expected profit ${rupees(r.expected_profit)}`)], {
    ...BASE_LAYOUT, title: { text: "Expected profit vs loan size", font: { size: 15 } },
    xaxis: loanAxis(),
    yaxis: yaxis({ title: { text: "expected profit" }, ...niceRupeeTicks(profit) }),
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: COLORS.muted, width: 1 } }],
  }, PLOT_CONFIG);
}

// ------------------------------------------------------------------ boot
function showError(msg) {
  const el = $("#load-error");
  el.textContent = msg;
  el.hidden = false;
}

function setPill(kind, text) {
  const pill = $("#mode-pill");
  pill.className = `pill ${kind}`;
  pill.textContent = text;
}

async function boot() {
  let res;
  try {
    res = await fetch("/api/meta");
  } catch {
    setPill("down", "API offline");
    return showError("Cannot reach the API. Start it with: .\\.venv\\Scripts\\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000");
  }
  meta = await res.json();
  if (!meta.ready) {
    setPill("down", "Model missing");
    return showError(meta.error);
  }
  setPill(meta.demo ? "demo" : "live", meta.demo ? "Demo mode" : "Live model");
  $("#demo-banner").hidden = !meta.demo;

  $("#kpi-threshold").textContent = pct(meta.threshold);
  $("#kpi-lgd").textContent = pct(meta.lgd, 0);
  $("#kpi-interest").textContent = pct(meta.interest, 0);
  $("#kpi-fixed").textContent = rupees(meta.fixed);

  const edu = $("#education");
  for (const e of meta.education) edu.append(new Option(e, e));

  const presetBox = $("#presets");
  for (const [name, values] of Object.entries(meta.presets)) {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.preset = name;
    b.append(name);
    const note = document.createElement("small");
    note.textContent = PRESET_NOTES[name] || "";
    b.append(note);
    b.addEventListener("click", () => {
      writeForm(values);
      presetBox.querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
      score();
    });
    presetBox.append(b);
  }
  presetBox.querySelector("button").click();   // start on "Typical", like app.py
}

boot();
