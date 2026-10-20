import streamlit as st

from utils import OUTCOME_HELP, OUTCOMES, diff_html, long_format, MODELS, pair_block, require_results, safe, sort_types

st.title("Error Explorer")
st.caption("Each row is one model's behaviour on one edited sentence pair. Leave a filter empty to include everything.")

df = require_results()
L = long_format(df)

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    models = c1.multiselect("Model", list(MODELS))
    kinds = c2.multiselect("Edit kind", sorted(L.edit_kind.unique()))
    types = c3.multiselect("Perturbation type", sort_types(L.perturbation_type.unique()))
    c4, c5, c6 = st.columns(3)
    edits = c4.multiselect("Edit (old -> new)", sorted(e for e in L.change.unique() if e))
    labels = c5.multiselect("Original gold label", sorted(L.original_label.unique()))
    outcomes = c6.multiselect("Outcome", OUTCOMES)
    query = st.text_input("Search in sentences", placeholder="e.g. BERT")
st.caption("Model confidence scores are not stored in the current results file, so a confidence filter is not available.")

f = L
for col, sel in [("model", models), ("edit_kind", kinds), ("perturbation_type", types),
                 ("change", edits), ("original_label", labels), ("outcome", outcomes)]:
    if sel:
        f = f[f[col].isin(sel)]
if query:
    q = query.lower()
    f = f[f.sentence1.str.lower().str.contains(q, regex=False)
          | f.sentence2_original.str.lower().str.contains(q, regex=False)
          | f.sentence2_new.str.lower().str.contains(q, regex=False)]
f = f.reset_index(drop=True)

bad = f.outcome.isin(["Flipped (inconsistent)", "Ignored the edit"]).sum() if len(f) else 0
m1, m2, m3 = st.columns(3)
m1.metric("Rows", len(f))
m2.metric("Unexpected behaviour", int(bad), help="Flipped on a meaning-preserving edit, or ignored a meaning-changing edit.")
m3.metric("Share unexpected", f"{bad / len(f) * 100:.1f}%" if len(f) else "N/A")

if f.empty:
    st.info("No rows match these filters.")
    st.stop()

st.markdown("#### Outcomes by edit type")
st.dataframe(f.groupby(["perturbation_type", "outcome"]).size().unstack(fill_value=0))

st.markdown("#### Matching rows")
show = ["model", "edit_kind", "perturbation_type", "change", "original_label", "pred_orig", "pred_new",
        "outcome", "sentence2_original", "sentence2_new"]
st.dataframe(f[show], hide_index=True)
export_cols = ["model", "edit_kind", "perturbation_type", "change", "review_status", "original_id", "sentence1",
               "sentence2_original", "sentence2_new", "original_label", "pred_orig", "pred_new",
               "prediction_changed", "outcome"]
st.download_button("Download filtered rows (CSV)", f[export_cols].to_csv(index=False).encode("utf-8-sig"),
                   file_name="error_explorer_filtered.csv", mime="text/csv")

st.markdown("#### Inspect an example")
limit = min(len(f), 300)
idx = st.selectbox("Choose an example", range(limit),
                   format_func=lambda i: f"{i + 1}. {f.loc[i, 'model']} | {f.loc[i, 'perturbation_type']} | {f.loc[i, 'change'][:45]}")
r = f.loc[idx]
old_html, new_html = diff_html(r.sentence2_original, r.sentence2_new)
st.markdown(pair_block("Sentence 1 (unchanged)", safe(r.sentence1))
            + pair_block("Sentence 2, original", old_html)
            + pair_block("Sentence 2, edited", new_html), unsafe_allow_html=True)

after = r.original_label if r.is_meaning_preserving else "not annotated"
g1, g2, g3, g4 = st.columns(4)
g1.metric("Gold label (original)", r.original_label)
g2.metric("Gold label (after edit)", after)
g3.metric("Prediction (original)", r.pred_orig)
g4.metric("Prediction (edited)", r.pred_new)
st.markdown(f"**Outcome: {r.outcome}.** {OUTCOME_HELP[r.outcome]}")
st.caption(f"Validation status: {r.review_status}. Edit: `{r.change}`")
