import pandas as pd
import joblib
from scipy.sparse import hstack

vec, clf = joblib.load("models/tfidf_lr.joblib")

def feats(s1, s2):
    a, b = vec.transform(s1), vec.transform(s2)
    return hstack([a, b, a.multiply(b)]).tocsr()

df = pd.read_csv("data/processed/perturbations_raw.csv")
# rows you marked valid="no" are dropped; blank rows are kept for now
df = df[df["valid"].astype(str).str.lower() != "no"].copy()

df["pred_orig"] = clf.predict(feats(df.sentence1, df.sentence2_original))
df["pred_new"] = clf.predict(feats(df.sentence1, df.sentence2_new))

# Make sure the flag is a real boolean column
df["is_meaning_preserving"] = df["is_meaning_preserving"].astype(str).str.lower() == "true"

keep = df[df["is_meaning_preserving"]].copy()
keep["consistent"] = keep["pred_orig"] == keep["pred_new"]
print("CONSISTENCY on meaning-preserving edits (higher = better):")
print(keep.groupby("perturbation_type")["consistent"].mean().round(3))
print("Overall:", round(keep["consistent"].mean(), 3))

change = df[~df["is_meaning_preserving"]].copy()
change["changed"] = change["pred_orig"] != change["pred_new"]
print("\nPREDICTION-CHANGE RATE on meaning-changing edits (higher = better):")
print(change.groupby("perturbation_type")["changed"].mean().round(3))
print("Overall:", round(change["changed"].mean(), 3))