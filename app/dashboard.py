import pandas as pd
import streamlit as st

st.set_page_config(page_title="NLI Robustness Benchmark", layout="wide")
st.title("NLI Robustness Benchmark")
st.caption("Which linguistic changes break the models? (TF-IDF vs SciBERT on SciNLI)")

df = pd.read_csv("data/processed/final_results.csv")
df["is_meaning_preserving"] = df["is_meaning_preserving"].astype(str).str.lower() == "true"

models = {"TF-IDF": "tfidf", "SciBERT": "scibert"}

pres = df[df.is_meaning_preserving].copy()
chg = df[~df.is_meaning_preserving].copy()

cons, sens = {}, {}
for name, m in models.items():
    cons[name] = (pres[f"{m}_pred_orig"] == pres[f"{m}_pred_new"]).groupby(pres["perturbation_type"]).mean()
    sens[name] = (chg[f"{m}_pred_orig"] != chg[f"{m}_pred_new"]).groupby(chg["perturbation_type"]).mean()
cons, sens = pd.DataFrame(cons), pd.DataFrame(sens)

# Top numbers
c1, c2, c3, c4 = st.columns(4)
c1.metric("TF-IDF test Macro-F1", "0.530")
c2.metric("SciBERT test Macro-F1", "0.776")
c3.metric("SciBERT negation sensitivity", f"{sens.loc['negation', 'SciBERT']:.2f}")
c4.metric("Rows checked", f"{len(pres)} + {len(chg)}")

left, right = st.columns(2)
with left:
    st.subheader("Meaning-preserving edits")
    st.caption("Consistency: prediction stayed the same (higher = better)")
    st.bar_chart(cons)
    st.dataframe(pres.groupby("perturbation_type").size().rename("n"))
with right:
    st.subheader("Meaning-changing edits")
    st.caption("Sensitivity: prediction changed (higher = model noticed the change)")
    st.bar_chart(sens)
    st.dataframe(chg.groupby("perturbation_type").size().rename("n"))

st.warning("Small samples: quantity n=13, causal n=18. Sensitivity is not accuracy; "
           "gold labels for meaning-changing edits were not annotated.")

# Examples
st.subheader("Browse examples")
f1, f2, f3 = st.columns(3)
model_name = f1.selectbox("Model", list(models.keys()))
kind = f2.selectbox("Edit kind", ["Meaning-changing", "Meaning-preserving"])
ptype = f3.selectbox("Type", ["all"] + sorted(df.perturbation_type.unique()))

m = models[model_name]
sub = chg if kind == "Meaning-changing" else pres
if ptype != "all":
    sub = sub[sub.perturbation_type == ptype]

sub = sub.copy()
sub["prediction_changed"] = sub[f"{m}_pred_orig"] != sub[f"{m}_pred_new"]
only = st.checkbox("Show only rows where the model did NOT behave well "
                   "(changing: prediction unchanged / preserving: prediction changed)")
if only:
    sub = sub[~sub.prediction_changed] if kind == "Meaning-changing" else sub[sub.prediction_changed]

st.write(f"{len(sub)} rows")
st.dataframe(sub[["perturbation_type", "change", "sentence2_original", "sentence2_new",
                  "original_label", f"{m}_pred_orig", f"{m}_pred_new"]],
             use_container_width=True)
