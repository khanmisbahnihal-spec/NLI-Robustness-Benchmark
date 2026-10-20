import streamlit as st

from utils import card, fmt, load_test_metrics, lookup, note, pct, require_results, summary_table

st.title("NLI Robustness Lab")
st.markdown("##### Do scientific NLI models notice when the meaning of a sentence changes?")

df = require_results()

st.markdown("### The research problem")
st.write(
    "Natural language inference (NLI) models decide how two sentences relate. SciNLI contains sentence pairs "
    "from scientific papers with four relations: entailment, contrasting, reasoning and neutral. "
    "This project edits the second sentence in small, controlled ways (negation, quantities, qualifiers, "
    "causal language) and checks whether model predictions react appropriately."
)
st.markdown("### Why test-set accuracy does not guarantee robustness")
st.write(
    "A model can score well on a held-out test set by relying on topic words and discourse cues. If it does, "
    "changing \"is\" to \"is not\" may leave its prediction untouched, even though the meaning has been reversed. "
    "A single accuracy number cannot reveal this, so we measure how predictions behave under controlled edits."
)

st.markdown("### Workflow")
cols = st.columns(4)
steps = [
    ("STEP 1", "Dataset", "SciNLI: 107,412 scientific sentence pairs, four labels."),
    ("STEP 2", "Perturbation engine", "Rule-based minimal edits, split into meaning-preserving and meaning-changing."),
    ("STEP 3", "Model evaluation", "TF-IDF + Logistic Regression and a fine-tuned SciBERT."),
    ("STEP 4", "Error analysis", "Break results down by edit type and inspect individual failures."),
]
for col, (step, title, body) in zip(cols, steps):
    col.markdown(card(title, body, step), unsafe_allow_html=True)

st.markdown("### Benchmark preview")
S = summary_table(df)
metrics = load_test_metrics()
c1, c2, c3, c4 = st.columns(4)
c1.metric("TF-IDF test Macro-F1", fmt(metrics.get("TF-IDF + LogReg")))
c2.metric("SciBERT test Macro-F1", fmt(metrics.get("SciBERT")))
cons, n_cons = lookup(S, "SciBERT", "Consistency")
c3.metric("SciBERT consistency", pct(cons), help=f"Share of meaning-preserving edits (n={n_cons}) where the prediction did not change.")
sens, n_neg = lookup(S, "SciBERT", "Sensitivity", "negation")
c4.metric("SciBERT negation sensitivity", pct(sens), help=f"Share of negation edits (n={n_neg}) where the prediction changed.")

note("Sensitivity is a proxy, not accuracy. Gold labels were not annotated for meaning-changing edits, because the "
     "correct label after an edit is often ambiguous. Sample sizes per edit type are small; see the Benchmark page "
     "for counts and confidence intervals.")

st.markdown("### Explore")
n1, n2, n3 = st.columns(3)
n1.page_link("views/benchmark.py", label="Benchmark", icon=":material/bar_chart:")
n2.page_link("views/error_explorer.py", label="Error Explorer", icon=":material/search:")
n3.page_link("views/methodology.py", label="Methodology", icon=":material/menu_book:")
