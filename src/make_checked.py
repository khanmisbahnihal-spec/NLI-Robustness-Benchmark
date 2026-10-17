import pandas as pd
import difflib
from pathlib import Path

src = Path("data/processed/perturbations_raw.csv")
dst = Path("data/processed/perturbations_checked.csv")

# Safety: never overwrite your manual work
if dst.exists():
    raise SystemExit("perturbations_checked.csv already exists. Not overwriting.")

df = pd.read_csv(src)

def diff(a, b):
    a_w, b_w = str(a).split(), str(b).split()
    sm = difflib.SequenceMatcher(None, a_w, b_w)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            parts.append(" ".join(a_w[i1:i2]) + " -> " + " ".join(b_w[j1:j2]))
    return " | ".join(parts)

df["change"] = [diff(a, b) for a, b in zip(df.sentence2_original, df.sentence2_new)]

# preserving rows first, then changing; easier to check in one go
df = df.sort_values(["is_meaning_preserving", "perturbation_type"],
                    ascending=[False, True]).reset_index(drop=True)

cols = ["original_id", "perturbation_type", "is_meaning_preserving", "change",
        "sentence1", "sentence2_original", "sentence2_new",
        "original_label", "gold_label", "valid"]
df[cols].to_csv(dst, index=False, encoding="utf-8-sig")
print("Created", dst, "with", len(df), "rows")