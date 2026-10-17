import pandas as pd
import joblib
from pathlib import Path
from scipy.sparse import hstack

D = Path("data/processed")
vec, clf = joblib.load("models/tfidf_lr.joblib")

def feats(s1, s2):
    a, b = vec.transform(s1), vec.transform(s2)
    return hstack([a, b, a.multiply(b)]).tocsr()

def is_true(s):
    return s.astype(str).str.strip().str.lower() == "true"

def is_yes(s):
    return s.astype(str).str.strip().str.lower() == "yes"

checked = pd.read_csv(D / "perturbations_checked.csv", encoding="utf-8-sig")
changing = pd.read_csv(D / "changing_100.csv", encoding="utf-8-sig")
sci = pd.read_csv(D / "scibert_predictions.csv")

checked["is_meaning_preserving"] = is_true(checked["is_meaning_preserving"])
sci["is_meaning_preserving"] = is_true(sci["is_meaning_preserving"])

# Only rows you marked valid = yes
pres = checked[checked["is_meaning_preserving"] & is_yes(checked["valid"])].copy()
chg = changing[is_yes(changing["valid"])].copy()
chg["is_meaning_preserving"] = False

print("Valid meaning-preserving rows:", len(pres))
print("Valid meaning-changing rows:  ", len(chg))
print("Still CHECK/blank in preserving:",
      (checked["is_meaning_preserving"] & ~checked["valid"].astype(str).str.lower().isin(["yes", "no"])).sum())

cols = ["original_id", "perturbation_type", "is_meaning_preserving", "change",
        "sentence1", "sentence2_original", "sentence2_new", "original_label"]
df = pd.concat([pres[cols], chg[cols]], ignore_index=True)

# Attach SciBERT predictions
keys = ["original_id", "perturbation_type", "is_meaning_preserving", "sentence2_new"]
sci_small = sci[keys + ["scibert_pred_orig", "scibert_pred_new"]].drop_duplicates(keys)
df = df.merge(sci_small, on=keys, how="left")
print("Rows without SciBERT predictions:", df["scibert_pred_new"].isna().sum())

# TF-IDF predictions
df["tfidf_pred_orig"] = clf.predict(feats(df.sentence1, df.sentence2_original))
df["tfidf_pred_new"] = clf.predict(feats(df.sentence1, df.sentence2_new))

df.to_csv(D / "final_results.csv", index=False, encoding="utf-8-sig")

def report(model):
    po, pn = f"{model}_pred_orig", f"{model}_pred_new"

    keep = df[df.is_meaning_preserving].copy()
    keep["consistent"] = keep[po] == keep[pn]
    keep["acc_orig"] = keep[po] == keep["original_label"]
    keep["acc_new"] = keep[pn] == keep["original_label"]
    t1 = keep.groupby("perturbation_type").agg(
        n=("consistent", "size"),
        consistency=("consistent", "mean"),
        acc_orig=("acc_orig", "mean"),
        acc_new=("acc_new", "mean"),
    )
    t1["acc_drop"] = t1["acc_orig"] - t1["acc_new"]

    ch = df[~df.is_meaning_preserving].copy()
    ch["changed"] = ch[po] != ch[pn]
    t2 = ch.groupby("perturbation_type").agg(n=("changed", "size"),
                                              sensitivity=("changed", "mean"))

    print("\n" + "=" * 60)
    print(model.upper())
    print("\nMeaning-preserving edits (consistency higher = better):")
    print(t1.round(3))
    print("Overall consistency:", round(keep["consistent"].mean(), 3))
    print("\nMeaning-changing edits (sensitivity = how often prediction changed):")
    print(t2.round(3))
    print("Overall sensitivity:", round(ch["changed"].mean(), 3))

report("tfidf")
report("scibert")
print("\nSaved data/processed/final_results.csv")
