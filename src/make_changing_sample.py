import pandas as pd

df = pd.read_csv("data/processed/perturbations_checked.csv", encoding="utf-8-sig")
ch = df[df["is_meaning_preserving"] == False]

# 25 random rows per type
sample = (ch.sample(frac=1, random_state=1)
            .groupby("perturbation_type").head(25))

cols = ["original_id", "perturbation_type", "change", "sentence1",
        "sentence2_original", "sentence2_new", "original_label"]
out = sample[cols].copy()
out["gold_label"] = ""
out["valid"] = ""
out.to_csv("data/processed/changing_100.csv", index=False, encoding="utf-8-sig")
print("Saved", len(out), "rows")