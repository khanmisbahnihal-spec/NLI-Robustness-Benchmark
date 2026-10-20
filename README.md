# NLI Robustness Benchmark

**Live dashboard:** https://nli-robustness-benchmark-75s26nnrzqlsf5yvutcdvb.streamlit.app/

A small benchmark that applies controlled, rule-based edits (negation, quantities, qualifiers, causal language) to scientific sentence pairs and tests whether NLI model predictions change appropriately.

It is built on [SciNLI](https://huggingface.co/datasets/tasksource/scinli): 107,412 scientific sentence pairs with four labels (`entailment`, `contrasting`, `reasoning`, `neutral`).

## Key finding

SciBERT reaches a test Macro-F1 of about 0.78, but it almost ignores negation. After inserting "not" into the second sentence, its prediction changed in only 1 of 23 valid cases. High accuracy on the original test set does not mean a model tracks these linguistic cues.

## Results

**Original test set (4,000 pairs)**

| Model | Macro-F1 |
|---|---|
| TF-IDF + Logistic Regression | 0.530 |
| SciBERT (40k training pairs, 2 epochs) | see `data/processed/test_metrics.json` |

**Meaning-changing edits: sensitivity** (share of cases where the prediction changed)

| Edit type | n | TF-IDF | SciBERT |
|---|---|---|---|
| Negation | 23 | 0.39 | 0.04 |
| Qualifier | 25 | 0.28 | 0.24 |
| Causal | 18 | 0.11 | 0.11 |
| Quantity | 13 | 0.08 | 0.08 |

**Meaning-preserving edits: consistency** is about 0.88 for TF-IDF and about 0.97 for SciBERT. Per-type values, sample sizes and 95% bootstrap intervals are on the Benchmark page of the dashboard.

Sensitivity is a proxy, not accuracy: gold labels were not annotated for meaning-changing edits because the correct label after an edit is often ambiguous.

## Dashboard pages

- **Overview**: problem statement, workflow and headline numbers.
- **Benchmark**: Macro-F1, consistency, sensitivity, accuracy before and after edits, with sample counts and bootstrap intervals.
- **Error Explorer**: filter failures by model, edit type, edit, label and outcome; inspect examples with highlighted changes; download filtered rows as CSV.
- **Methodology**: dataset, models, rules, validation, metric definitions, seeds and limitations.

## Project structure

```
NLI-Robustness-Benchmark/
├── app/
│   ├── dashboard.py            # entry point (navigation)
│   ├── utils.py                # data loading, metrics, charts
│   ├── views/                  # overview, benchmark, error_explorer, methodology
│   └── requirements.txt        # dependencies for the deployed app
├── .streamlit/config.toml      # theme
├── data/processed/             # perturbation files, predictions, final results
├── notebooks/train_scibert_colab.py
├── src/                        # baseline, perturbation, evaluation scripts
├── requirements.txt
└── README.md
```

## Setup and reproduction

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -r requirements.txt

python src/baseline_tfidf.py     # TF-IDF baseline, saves models/tfidf_lr.joblib
python src/perturbations.py      # creates data/processed/perturbations_raw.csv
# train SciBERT on Colab: notebooks/train_scibert_colab.py -> scibert_predictions.csv
python src/evaluate_final.py     # writes data/processed/final_results.csv
streamlit run app/dashboard.py
```

`perturbations.py` overwrites `perturbations_raw.csv`. The manually reviewed files (`perturbations_checked.csv`, `changing_100.csv`) are kept separate for this reason. The repository already contains the SciBERT predictions, so the evaluation and dashboard run without retraining.

## Limitations

- Small samples, especially meaning-changing quantity (n=13) and causal (n=18) edits.
- Sensitivity is not accuracy: an unchanged prediction is not necessarily wrong, and a changed one is not necessarily right.
- Regex edits can be awkward; the meaning-preserving set was filtered and spot-checked, not fully reviewed by hand.
- TF-IDF has about 50% accuracy on original pairs, so its higher sensitivity reflects instability rather than understanding.
- SciBERT was trained on a 40k subset for 2 epochs. The checkpoint revision was not pinned.
- Leakage between related pairs from the same source paper was not checked.
- Confidence scores and full test-set predictions are not stored, so confusion matrices are not shown.

## Dataset citation

Sadat, M. and Caragea, C. (2022). *SciNLI: A Corpus for Natural Language Inference on Scientific Text.* ACL 2022.
