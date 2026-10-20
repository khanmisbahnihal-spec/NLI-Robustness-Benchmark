import pandas as pd
import streamlit as st

from utils import (MODELS, TYPE_ORDER, accuracy_shift, fmt, grouped_bar, load_test_metrics, note, rate_table,
                   require_results, summary_table)

st.title("Benchmark")
st.caption("All numbers on this page are computed from data/processed/final_results.csv.")

df = require_results()
S = summary_table(df)

# ---------------------------------------------------------------- original test set
st.markdown("### 1. Original test set")
metrics = load_test_metrics()
c1, c2 = st.columns(2)
for col, name in zip((c1, c2), MODELS):
    col.metric(f"{name} Macro-F1", fmt(metrics.get(name)), help="SciNLI test set, 4,000 pairs, four classes.")
note("Confusion matrices: <b>N/A</b>. Predictions for the full test set are not stored in this repository yet, "
     "so a confusion matrix cannot be drawn from real outputs.")

# ---------------------------------------------------------------- preserving
st.markdown("### 2. Meaning-preserving edits: consistency")
st.caption("Synonym-style swaps. The prediction should stay the same. Higher is better. Whiskers show a 95% bootstrap interval.")
pres = S[S.metric == "Consistency"]
if pres.empty:
    st.info("No meaning-preserving results available.")
else:
    st.altair_chart(grouped_bar(pres, "Consistency"))
    st.dataframe(rate_table(pres), hide_index=True)

st.markdown("#### Accuracy before and after the edit")
st.caption("For meaning-preserving edits the gold label stays the same, so accuracy before and after is comparable. "
           "This is a subset of the test set, so these values differ from the Macro-F1 above.")
A = accuracy_shift(df)
if A.empty:
    st.info("Not available.")
else:
    order = {t: i for i, t in enumerate(TYPE_ORDER)}
    A = A.assign(_o=A.perturbation_type.map(lambda x: 99 if x == "overall" else order.get(x, 50))
                 ).sort_values(["_o", "model"])
    st.dataframe(pd.DataFrame({
        "Edit type": A.perturbation_type, "Model": A.model, "n": A.n,
        "Accuracy (original)": [f"{v * 100:.1f}%" for v in A.acc_original],
        "Accuracy (edited)": [f"{v * 100:.1f}%" for v in A.acc_perturbed],
        "Change (points)": [f"{v:+.1f}" for v in A.change_pp],
    }), hide_index=True)

# ---------------------------------------------------------------- changing
st.markdown("### 3. Meaning-changing edits: sensitivity")
st.caption("Edits that change the meaning. Sensitivity is the share of cases where the prediction changed. "
           "A model that notices the edit should react, but a reaction is not proof of a correct label.")
chg = S[S.metric == "Sensitivity"]
if chg.empty:
    st.info("No meaning-changing results available.")
else:
    st.altair_chart(grouped_bar(chg, "Sensitivity"))
    st.dataframe(rate_table(chg), hide_index=True)
    small = chg[(chg.perturbation_type != "overall") & (chg.n < 30)]
    if not small.empty:
        note("Small samples: " + ", ".join(sorted({f"{t} (n={int(n)})" for t, n in zip(small.perturbation_type, small.n)}))
             + ". Treat these rates as indicative only; the intervals are wide.", "warn")

with st.expander("Metric definitions"):
    st.markdown(
        "- **Macro-F1**: unweighted mean of the per-class F1 scores on the original SciNLI test set.\n"
        "- **Consistency**: share of meaning-preserving edits where the prediction on the edited pair equals the "
        "prediction on the original pair. It measures stability, not correctness.\n"
        "- **Sensitivity**: share of meaning-changing edits where the prediction changed. It measures whether the "
        "model reacts, not whether it reacts correctly.\n"
        "- **Accuracy change**: accuracy on the edited pair minus accuracy on the original pair, against the "
        "original gold label, for meaning-preserving edits only.\n"
        "- **95% interval**: percentile bootstrap, 2,000 resamples, fixed seed."
    )
