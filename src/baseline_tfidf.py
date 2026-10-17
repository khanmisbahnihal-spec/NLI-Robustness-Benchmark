import pandas as pd
from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import joblib

ds = load_dataset("tasksource/scinli")
tr, va, te = [ds[s].to_pandas() for s in ("train", "validation", "test")]

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
vec.fit(pd.concat([tr.sentence1, tr.sentence2]))

def feats(df):
    a, b = vec.transform(df.sentence1), vec.transform(df.sentence2)
    return hstack([a, b, a.multiply(b)]).tocsr()

clf = LogisticRegression(max_iter=1000, C=2.0)
clf.fit(feats(tr), tr.label)

pred = clf.predict(feats(te))
print("Macro-F1:", f1_score(te.label, pred, average="macro"))
print(classification_report(te.label, pred))

from pathlib import Path
Path("models").mkdir(exist_ok=True)
joblib.dump((vec, clf), "models/tfidf_lr.joblib")
