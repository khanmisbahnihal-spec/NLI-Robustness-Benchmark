import pandas as pd
import streamlit as st

from utils import note, require_results

st.title("Methodology and reproducibility")

df = require_results()
n_pres = int(df.is_meaning_preserving.sum())
n_chg = int((~df.is_meaning_preserving).sum())

st.markdown("### Dataset")
st.markdown(
    "- **Source:** SciNLI (Sadat and Caragea, 2022), loaded as `tasksource/scinli` from Hugging Face. "
    "Check the dataset card for the licence.\n"
    "- **Size:** 107,412 sentence pairs. Splits used: train 101,412, validation 2,000, test 4,000.\n"
    "- **Labels:** four classes: `contrasting`, `entailment`, `neutral`, `reasoning`. The training split has "
    "25,353 pairs per class (balanced).\n"
    "- **Held-out data:** perturbations are generated from the test split only. No perturbed example was used for training."
)

st.markdown("### Models")
st.markdown(
    "**Baseline: TF-IDF + Logistic Regression.** TF-IDF with unigrams and bigrams (`min_df=3`, `max_features=200,000`, "
    "`sublinear_tf=True`) fitted on all training sentences. Features are the TF-IDF vectors of both sentences and their "
    "element-wise product. Logistic Regression with `C=2.0`, `max_iter=1000`.\n\n"
    "**Pretrained transformer: SciBERT.** `allenai/scibert_scivocab_uncased` fine-tuned for four-class sentence-pair "
    "classification on a random 40,000-pair subset of the training set (seed 42), 2 epochs, learning rate 2e-5, batch size 32, "
    "max length 128, fp16, on a Google Colab T4 GPU. The Hugging Face checkpoint revision was not pinned."
)

st.markdown("### Perturbation rules")
st.write("Rules are regular expressions applied to sentence 2 of test pairs. For each type, the first matching rule is applied once.")
preserving = pd.DataFrame([
    ("qualifier", "significantly -> notably; clearly -> evidently; generally -> typically"),
    ("quantity", "about (before a number) -> roughly; approximately -> roughly; many <-> numerous"),
    ("causal", "leads to <-> results in; because of <-> due to; causes -> brings about; caused by -> brought about by"),
    ("generic", "Therefore -> Thus; However -> Nevertheless; show(s) -> demonstrate(s)"),
], columns=["Type", "Meaning-preserving rules"])
changing = pd.DataFrame([
    ("negation", "is -> is not; are -> are not; increases -> does not increase; improves -> does not improve"),
    ("quantity", "all -> some; most -> few; many -> few"),
    ("qualifier", "always -> sometimes; may -> must; can -> cannot"),
    ("causal", "causes -> is associated with; leads to / results in -> is unrelated to; because of / due to -> despite; because -> although"),
], columns=["Type", "Meaning-changing rules"])
st.dataframe(preserving, hide_index=True)
st.dataframe(changing, hide_index=True)

st.markdown("### Validation procedure")
st.markdown(
    f"- **Meaning-preserving ({n_pres} valid rows used):** a filter flagged risky patterns (for example, \"causes\" used as a noun). "
    "Flagged rows were checked by hand; the remaining rows were accepted after a random spot-check, not a full review.\n"
    f"- **Meaning-changing ({n_chg} valid rows used):** 100 rows were sampled (25 per type) and read manually. Rows with broken grammar "
    "or unclear meaning were removed.\n"
    "- **Labels:** meaning-preserving edits keep the original gold label. Meaning-changing edits have no gold label, because the "
    "correct SciNLI label after an edit is often ambiguous."
)

st.markdown("### Metrics")
st.markdown(
    "- **Macro-F1** on the original test set.\n"
    "- **Consistency** on meaning-preserving edits: the prediction on the edited pair equals the prediction on the original pair.\n"
    "- **Sensitivity** on meaning-changing edits: the prediction changed. This is a proxy, not accuracy.\n"
    "- **95% bootstrap intervals**: 2,000 resamples, seed 42.\n"
    "- **Outcome labels** in the Error Explorer: Stable, Flipped, Reacted to edit, Ignored the edit."
)

st.markdown("### Seeds and settings")
st.markdown(
    "Training subset shuffle: seed 42. Perturbation sampling: seed 42, up to 100 rows per type. Manual-review sample: seed 1, "
    "25 rows per type. Bootstrap: seed 42."
)

st.markdown("### Limitations")
note("<b>Please read before citing any number.</b>", "warn")
st.markdown(
    "- Sample sizes are small, especially for meaning-changing quantity and causal edits. Intervals are wide.\n"
    "- Sensitivity is not accuracy. A prediction that did not change is not necessarily wrong, because a rule-based edit may leave "
    "the relation between the two sentences intact, and a prediction that changed is not necessarily right.\n"
    "- Negation edits are inserted at the first matching word, which may sit in a clause that is not central to the sentence-pair relation.\n"
    "- Rule-based edits can create awkward sentences. Invalid examples were removed, but the meaning-preserving set was not fully reviewed by hand.\n"
    "- TF-IDF has about 50% accuracy on the original pairs, so its higher sensitivity reflects instability, not understanding.\n"
    "- SciBERT was trained on a subset for 2 epochs to fit a free GPU session. Results may differ with full training.\n"
    "- Leakage between related sentence pairs from the same source paper across splits was not checked.\n"
    "- Model confidence scores and full test-set predictions are not stored, so confidence analysis and confusion matrices are unavailable."
)

st.markdown("### Reproduce the results")
st.code(
    "python -m venv .venv\n"
    ".venv\\Scripts\\activate\n"
    "pip install -r requirements.txt\n"
    "python src/baseline_tfidf.py          # TF-IDF baseline, saves models/tfidf_lr.joblib\n"
    "python src/perturbations.py           # creates data/processed/perturbations_raw.csv\n"
    "# train SciBERT on Colab with notebooks/train_scibert_colab.py\n"
    "python src/evaluate_final.py          # writes data/processed/final_results.csv\n"
    "streamlit run app/dashboard.py", language="bash")
