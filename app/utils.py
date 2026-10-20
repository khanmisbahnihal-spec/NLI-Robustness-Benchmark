"""Shared helpers for the NLI Robustness Lab dashboard."""
from __future__ import annotations

import difflib
import html
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "data" / "processed" / "final_results.csv"
METRICS_JSON = ROOT / "data" / "processed" / "test_metrics.json"
REPO_URL = "https://github.com/khanmisbahnihal-spec/NLI-Robustness-Benchmark"

MODELS = {"TF-IDF + LogReg": "tfidf", "SciBERT": "scibert"}
COLORS = {"TF-IDF + LogReg": "#94a3b8", "SciBERT": "#0f2a4a"}
TYPE_ORDER = ["negation", "quantity", "qualifier", "causal", "generic"]
OUTCOMES = ["Stable (expected)", "Flipped (inconsistent)", "Reacted to edit", "Ignored the edit"]
OUTCOME_HELP = {
    "Stable (expected)": "The edit should not change the meaning, and the prediction stayed the same.",
    "Flipped (inconsistent)": "The edit should not change the meaning, but the prediction changed.",
    "Reacted to edit": ("The edit changes the meaning and the prediction changed. This shows sensitivity, "
                        "not necessarily correctness: no gold label exists for the edited pair."),
    "Ignored the edit": ("The edit changes the meaning but the prediction stayed the same. This can be a model "
                         "weakness, or the edit may not change the relation between the two sentences."),
}
REQUIRED = {
    "original_id", "perturbation_type", "is_meaning_preserving", "sentence1", "sentence2_original",
    "sentence2_new", "original_label", "tfidf_pred_orig", "tfidf_pred_new",
    "scibert_pred_orig", "scibert_pred_new",
}

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1150px;}
h1, h2, h3 {color: #0f2a4a; letter-spacing: -0.01em;}
.card {border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem 1.1rem; background: #ffffff; height: 100%;}
.card h4 {margin: 0 0 .35rem 0; color: #0f2a4a; font-size: 1rem;}
.card p {margin: 0; color: #475569; font-size: .92rem; line-height: 1.45;}
.step-n {color: #2563eb; font-weight: 700; font-size: .78rem; letter-spacing: .06em;}
.note {border-left: 4px solid #2563eb; background: #f1f5f9; padding: .7rem 1rem; border-radius: 4px;
       color: #334155; font-size: .93rem; margin: .5rem 0 1rem 0;}
.warn {border-left: 4px solid #b45309; background: #fffbeb; padding: .7rem 1rem; border-radius: 4px;
       color: #78350f; font-size: .93rem; margin: .5rem 0 1rem 0;}
.pair {border: 1px solid #e2e8f0; border-radius: 8px; padding: .8rem 1rem; background: #f8fafc;
       color: #0f172a; font-size: .95rem; line-height: 1.55; margin-bottom: .6rem;}
.pair .lbl {font-size: .72rem; font-weight: 700; letter-spacing: .06em; color: #64748b;
            text-transform: uppercase; margin-bottom: .25rem;}
mark.del {background: #fee2e2; color: #7f1d1d; text-decoration: line-through; padding: 0 2px; border-radius: 3px;}
mark.add {background: #dbeafe; color: #1e3a8a; padding: 0 2px; border-radius: 3px;}
[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: .8rem 1rem;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def safe(text) -> str:
    """Escape text for HTML and stop Markdown from treating $ as math."""
    return html.escape(str(text)).replace("$", "&#36;")


def card(title: str, body: str, step: str | None = None) -> str:
    head = f'<div class="step-n">{safe(step)}</div>' if step else ""
    return f'<div class="card">{head}<h4>{safe(title)}</h4><p>{safe(body)}</p></div>'


def note(text: str, kind: str = "note") -> None:
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV, encoding="utf-8-sig")
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError("final_results.csv is missing columns: " + ", ".join(sorted(missing)))
    df["is_meaning_preserving"] = df["is_meaning_preserving"].astype(str).str.strip().str.lower().eq("true")
    df["edit_kind"] = np.where(df["is_meaning_preserving"], "Meaning-preserving", "Meaning-changing")
    df["review_status"] = np.where(df["is_meaning_preserving"],
                                   "Rule-filtered and spot-checked", "Manually reviewed")
    if "change" not in df.columns:
        df["change"] = ""
    df["change"] = df["change"].fillna("").astype(str)
    return df


def require_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        st.error("`data/processed/final_results.csv` was not found. Run `python src/evaluate_final.py` "
                 "and push the file to GitHub.")
        st.stop()
    try:
        return load_results()
    except ValueError as err:
        st.error(str(err))
        st.stop()


def load_test_metrics() -> dict:
    """Original test-set Macro-F1 per model, from data/processed/test_metrics.json."""
    try:
        data = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: float(v) for k, v in data.items() if k in MODELS and isinstance(v, (int, float))}


def fmt(x, digits: int = 3) -> str:
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def pct(x) -> str:
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"


@st.cache_data(show_spinner=False)
def long_format(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (example, model) with the behaviour of that model."""
    parts = []
    for name, key in MODELS.items():
        d = df.copy()
        d["model"] = name
        d["pred_orig"] = d[f"{key}_pred_orig"]
        d["pred_new"] = d[f"{key}_pred_new"]
        d["prediction_changed"] = d["pred_orig"] != d["pred_new"]
        pres = d["is_meaning_preserving"]
        d["outcome"] = np.select(
            [pres & ~d["prediction_changed"], pres & d["prediction_changed"], ~pres & d["prediction_changed"]],
            OUTCOMES[:3], default=OUTCOMES[3])
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 42):
    """Mean and 95% percentile bootstrap interval of a 0/1 array."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


@st.cache_data(show_spinner=False)
def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Consistency (meaning-preserving) and sensitivity (meaning-changing) with bootstrap CIs."""
    L = long_format(df)
    rows = []
    for (model, kind), g in L.groupby(["model", "edit_kind"]):
        metric = "Consistency" if kind == "Meaning-preserving" else "Sensitivity"
        for ptype, gg in list(g.groupby("perturbation_type")) + [("overall", g)]:
            vals = ~gg["prediction_changed"] if metric == "Consistency" else gg["prediction_changed"]
            m, lo, hi = bootstrap_ci(vals.to_numpy())
            rows.append(dict(model=model, edit_kind=kind, metric=metric, perturbation_type=ptype,
                             n=len(gg), value=m, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def accuracy_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy against the (unchanged) gold label before and after a meaning-preserving edit."""
    L = long_format(df)
    L = L[L["is_meaning_preserving"]]
    rows = []
    for model, gm in L.groupby("model"):
        for ptype, g in list(gm.groupby("perturbation_type")) + [("overall", gm)]:
            a0 = float((g["pred_orig"] == g["original_label"]).mean())
            a1 = float((g["pred_new"] == g["original_label"]).mean())
            rows.append(dict(model=model, perturbation_type=ptype, n=len(g),
                             acc_original=a0, acc_perturbed=a1, change_pp=(a1 - a0) * 100))
    return pd.DataFrame(rows)


def lookup(S: pd.DataFrame, model: str, metric: str, ptype: str = "overall"):
    r = S[(S.model == model) & (S.metric == metric) & (S.perturbation_type == ptype)]
    return (float(r.iloc[0].value), int(r.iloc[0].n)) if len(r) else (None, 0)


def sort_types(types):
    types = list(types)
    return sorted(types, key=lambda t: TYPE_ORDER.index(t) if t in TYPE_ORDER else 99)


def rate_table(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["_o"] = t["perturbation_type"].map(lambda x: 99 if x == "overall" else (TYPE_ORDER.index(x) if x in TYPE_ORDER else 50))
    t = t.sort_values(["_o", "model"])
    return pd.DataFrame({
        "Edit type": t["perturbation_type"], "Model": t["model"], "n": t["n"],
        "Rate": [pct(v) for v in t["value"]],
        "95% CI": [f"{lo * 100:.1f}% to {hi * 100:.1f}%" for lo, hi in zip(t["ci_low"], t["ci_high"])],
    })


def grouped_bar(t: pd.DataFrame, y_title: str) -> alt.Chart:
    order = [x for x in TYPE_ORDER + ["overall"] if x in set(t["perturbation_type"])]
    base = alt.Chart(t).encode(
        x=alt.X("perturbation_type:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("model:N"),
    )
    bars = base.mark_bar().encode(
        y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title=y_title, axis=alt.Axis(format="%")),
        color=alt.Color("model:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["model", "perturbation_type", alt.Tooltip("value:Q", format=".1%"), "n",
                 alt.Tooltip("ci_low:Q", format=".1%"), alt.Tooltip("ci_high:Q", format=".1%")],
    )
    err = base.mark_rule(color="#334155").encode(y="ci_low:Q", y2="ci_high:Q")
    return (bars + err).properties(height=300)


def diff_html(a: str, b: str):
    """Word-level diff of two sentences as HTML (removed words red, added words blue)."""
    aw, bw = str(a).split(), str(b).split()
    left, right = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, aw, bw).get_opcodes():
        la, rb = " ".join(aw[i1:i2]), " ".join(bw[j1:j2])
        if tag == "equal":
            left.append(safe(la))
            right.append(safe(rb))
        else:
            if la:
                left.append(f'<mark class="del">{safe(la)}</mark>')
            if rb:
                right.append(f'<mark class="add">{safe(rb)}</mark>')
    return " ".join(left), " ".join(right)


def pair_block(label: str, body_html: str) -> str:
    return f'<div class="pair"><div class="lbl">{safe(label)}</div>{body_html}</div>'
