# SciBERT fine-tuning on SciNLI + predictions on the perturbation set.
# Run on Google Colab with a GPU (Runtime -> Change runtime type -> T4 GPU).
# Before running, upload data/processed/perturbations_raw.csv to the Colab Files panel.
#
# In Colab, run each "CELL" block below in its own cell.

# ---------------------------------------------------------------- CELL 1
# !pip install -q transformers datasets accelerate scikit-learn

import pandas as pd, numpy as np
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import f1_score, classification_report

MODEL = "allenai/scibert_scivocab_uncased"
labels = ["contrasting", "entailment", "neutral", "reasoning"]
l2i = {l: i for i, l in enumerate(labels)}

ds = load_dataset("tasksource/scinli")
tr = ds["train"].to_pandas().sample(frac=1, random_state=42).head(40000)  # subset for speed
va = ds["validation"].to_pandas()
te = ds["test"].to_pandas()

tok = AutoTokenizer.from_pretrained(MODEL)

def encode(batch):
    return tok(batch["sentence1"], batch["sentence2"], truncation=True, max_length=128)

def to_ds(df, with_labels=True):
    d = {"sentence1": df.sentence1.tolist(), "sentence2": df.sentence2.tolist()}
    if with_labels:
        d["labels"] = [l2i[x] for x in df.label]
    return Dataset.from_dict(d).map(encode, batched=True)

train_ds, val_ds, test_ds = to_ds(tr), to_ds(va), to_ds(te)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=4)

def metrics(p):
    return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}

args = TrainingArguments(
    output_dir="scibert_out", num_train_epochs=2,
    per_device_train_batch_size=32, per_device_eval_batch_size=64,
    learning_rate=2e-5, fp16=True, eval_strategy="epoch",
    save_strategy="no", report_to="none",
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
trainer.train()

# Macro-F1 on the original test set
out = trainer.predict(test_ds)
pred = out.predictions.argmax(-1)
print("SciBERT test Macro-F1:", f1_score(out.label_ids, pred, average="macro"))
print(classification_report(out.label_ids, pred, target_names=labels))

# ---------------------------------------------------------------- CELL 2
df = pd.read_csv("perturbations_raw.csv")
df = df[df["valid"].astype(str).str.lower() != "no"].copy()
df["is_meaning_preserving"] = df["is_meaning_preserving"].astype(str).str.lower() == "true"

def predict(s1, s2):
    d = Dataset.from_dict({"sentence1": list(s1), "sentence2": list(s2)}).map(encode, batched=True)
    return trainer.predict(d).predictions.argmax(-1)

df["pred_orig"] = predict(df.sentence1, df.sentence2_original)
df["pred_new"] = predict(df.sentence1, df.sentence2_new)

keep = df[df.is_meaning_preserving].copy()
keep["consistent"] = keep.pred_orig == keep.pred_new
print("CONSISTENCY (preserving):")
print(keep.groupby("perturbation_type")["consistent"].mean().round(3))
print("Overall:", round(keep.consistent.mean(), 3))

change = df[~df.is_meaning_preserving].copy()
change["changed"] = change.pred_orig != change.pred_new
print("\nPREDICTION-CHANGE RATE (changing):")
print(change.groupby("perturbation_type")["changed"].mean().round(3))
print("Overall:", round(change.changed.mean(), 3))

# ---------------------------------------------------------------- CELL 3
# Save predictions and download them to data/processed/scibert_predictions.csv
from google.colab import files
out_df = pd.concat([keep, change]).sort_index()
out_df.rename(columns={"pred_orig": "scibert_pred_orig",
                       "pred_new": "scibert_pred_new"}, inplace=True)
out_df["scibert_pred_orig"] = out_df["scibert_pred_orig"].map(dict(enumerate(labels)))
out_df["scibert_pred_new"] = out_df["scibert_pred_new"].map(dict(enumerate(labels)))
out_df.to_csv("scibert_predictions.csv", index=False)
files.download("scibert_predictions.csv")
