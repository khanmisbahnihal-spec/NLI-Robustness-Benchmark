import re
import random
import pandas as pd
from datasets import load_dataset
from pathlib import Path

random.seed(42)

# (pattern, replacement) pairs. Word boundaries (\b) ensure whole-word match only.
# Meaning-preserving swaps
PRESERVING = {
    "qualifier": [
        (r"\bsignificantly\b", "notably"),
        (r"\bclearly\b", "evidently"),
        (r"\bgenerally\b", "typically"),
    ],
        "quantity": [
        (r"\babout(?= \d)", "roughly"),
        (r"\bapproximately\b", "roughly"),
        (r"\bmany\b", "numerous"),
        (r"\bnumerous\b", "many"),
    ],
    "causal": [
        (r"\bleads to\b", "results in"),
        (r"\bbecause of\b", "due to"),
        (r"\bcauses\b", "brings about"),
        (r"\bresults in\b", "leads to"),
        (r"\bdue to\b", "because of"),
        (r"\bcaused by\b", "brought about by"),
    ],
    "generic": [
        (r"\bTherefore\b", "Thus"),
        (r"\bHowever\b", "Nevertheless"),
        (r"\bshow\b", "demonstrate"),
        (r"\bshows\b", "demonstrates"),
    ],
}

# Meaning-changing swaps (gold label needs manual check)
CHANGING = {
    "negation": [
        (r"\bis\b", "is not"),
        (r"\bare\b", "are not"),
        (r"\bincreases\b", "does not increase"),
        (r"\bimproves\b", "does not improve"),
    ],
    "quantity": [
        (r"\ball\b", "some"),
        (r"\bmost\b", "few"),
        (r"\bmany\b", "few"),
    ],
    "qualifier": [
        (r"\balways\b", "sometimes"),
        (r"\bmay\b", "must"),
        (r"\bcan\b", "cannot"),
    ],
    "causal": [
        (r"\bcauses\b", "is associated with"),
        (r"\bleads to\b", "is unrelated to"),
        (r"\bbecause of\b", "despite"),
        (r"\bresults in\b", "is unrelated to"),
        (r"\bdue to\b", "despite"),
        (r"\bbecause\b", "although"),
    ],
}


def apply_edit(text, rules):
    """Apply the first matching rule once. Returns new text or None."""
    for pattern, repl in rules:
        if re.search(pattern, text):
            return re.sub(pattern, repl, text, count=1)
    return None


def generate(df, rule_dict, preserving):
    rows = []
    for _, r in df.iterrows():
        for ptype, rules in rule_dict.items():
            new_s2 = apply_edit(r.sentence2, rules)
            if new_s2 and new_s2 != r.sentence2:
                rows.append({
                    "original_id": r.id,
                    "perturbation_type": ptype,
                    "is_meaning_preserving": preserving,
                    "sentence1": r.sentence1,
                    "sentence2_original": r.sentence2,
                    "sentence2_new": new_s2,
                    "original_label": r.label,
                    # preserving: label stays same. changing: fill manually.
                    "gold_label": r.label if preserving else "",
                })
    return pd.DataFrame(rows)


def main():
    ds = load_dataset("tasksource/scinli")
    test = ds["test"].to_pandas()

    keep = generate(test, PRESERVING, True)
    change = generate(test, CHANGING, False)

    print("Meaning-preserving candidates:")
    print(keep.perturbation_type.value_counts())
    print("\nMeaning-changing candidates:")
    print(change.perturbation_type.value_counts())

    # Sample up to 100 per type so manual checking stays feasible
    def sample(df):
        # shuffle first, then keep up to 100 rows per type (keeps all columns)
        return (df.sample(frac=1, random_state=42)
                  .groupby("perturbation_type")
                  .head(100))

    out = pd.concat([sample(keep), sample(change)]).reset_index(drop=True)
    out["valid"] = ""  # you fill yes/no during manual check

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out.to_csv("data/processed/perturbations_raw.csv", index=False)
    print(f"\nSaved {len(out)} examples to data/processed/perturbations_raw.csv")


if __name__ == "__main__":
    main()