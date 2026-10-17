import pandas as pd
import re

path = "data/processed/perturbations_checked.csv"
df = pd.read_csv(path, encoding="utf-8-sig")
df["valid"] = df["valid"].astype("object")

# Patterns that often break the sentence after the edit
RISKY = [
    r"\b(the|of|its|their|main|possible|these|those)\s+causes\b",   # causes as a noun
    r"\bcauses\s+of\b",
    r"\b(the|these|our|their|experimental|empirical|results)\s+results\s+in\b",
    r"\b(a|the|this|that)\s+shows?\b",                              # show as a noun
]

def is_risky(row):
    return any(re.search(p, str(row.sentence2_original), flags=re.I) for p in RISKY)

mask = (df["is_meaning_preserving"] == True) & (df["valid"].isna() | (df["valid"].astype(str).str.strip() == ""))
for i in df[mask].index:
    df.at[i, "valid"] = "CHECK" if is_risky(df.loc[i]) else "yes"

df.to_csv(path, index=False, encoding="utf-8-sig")
print("Auto-filled preserving rows.")
print(df[df.is_meaning_preserving == True]["valid"].value_counts())