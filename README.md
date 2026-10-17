# NLI Robustness Benchmark

A small benchmark that applies controlled, rule-based edits (negation, quantities, qualifiers, causal language) to scientific sentence pairs and tests whether NLI model predictions change appropriately.

Built on [SciNLI](https://huggingface.co/datasets/tasksource/scinli), a scientific natural language inference dataset with 107,412 sentence pairs and four labels: `entailment`, `contrasting`, `reasoning`, `neutral`.

## Key finding

SciBERT reaches a test Macro-F1 of 0.776, but it almost ignores negation: after inserting "not" into the second sentence, its prediction changed in only **1 of 23** valid cases (4.3%). High accuracy on the original test set does not mean the model tracks these linguistic cues.

## Results

### Original test set (4,000 pairs)

| Model | Macro-F1 |
|---|---|
| TF-IDF + Logistic Regression | 0.530 |
| SciBERT (fine-tuned on 40k training pairs, 2 epochs) | 0.776 |

### Meaning-preserving edits (consistency, higher is better)

Synonym-style swaps (e.g. `significantly -> notably`, `leads to -> results in`). The prediction should stay the same.

| Edit type | n | TF-IDF | SciBERT |
|---|---|---|---|
| Causal | 99 | 0.909 | 0.960 |
| Generic | 90 | 0.944 | 0.978 |
| Qualifier | 53 | 0.811 | 0.981 |
| Quantity | 100 | 0.840 | 0.980 |
| **Overall** | 342 | 0.883 | 0.974 |

### Meaning-changing edits (sensitivity)

Edits that change the meaning (e.g. `is -> is not`, `all -> some`, `can -> cannot`, `because -> although`). Sensitivity is the share of cases where the model's prediction changed.

| Edit type | n | TF-IDF | SciBERT |
|---|---|---|---|
| Negation | 23 | 0.391 | 0.043 |
| Qualifier | 25 | 0.280 | 0.240 |
| Causal | 18 | 0.111 | 0.111 |
| Quantity | 13 | 0.077 | 0.077 |
| **Overall** | 79 | 0.241 | 0.127 |

> The numbers above are from one run. Re-run `src/evaluate_final.py` to reproduce them exactly.

## How it works

1. **Baselines**
   - TF-IDF (1-2 grams) features of both sentences plus their element-wise product, fed to Logistic Regression.
   - SciBERT (`allenai/scibert_scivocab_uncased`) fine-tuned for 4-class sentence-pair classification (trained on Google Colab, see `notebooks/train_scibert_colab.py`).
2. **Perturbations** (`src/perturbations.py`): regex rules applied to `sentence2` of test pairs.
   - *Meaning-preserving:* `significantly -> notably`, `approximately -> roughly`, `leads to -> results in`, `Therefore -> Thus`, ...
   - *Meaning-changing:* `is -> is not`, `all -> some`, `most -> few`, `always -> sometimes`, `may -> must`, `can -> cannot`, `causes -> is associated with`, `because -> although`, ...
   - Up to 100 examples per type are sampled.
3. **Validation**
   - Meaning-preserving rows: a filter flags risky patterns (e.g. "causes" as a noun) and the flagged rows were checked by hand; the rest were spot-checked.
   - Meaning-changing rows: 100 sampled rows were read manually and rows with broken grammar were removed (79 kept).
4. **Evaluation** (`src/evaluate_final.py`): consistency on meaning-preserving edits and sensitivity on meaning-changing edits, per edit type, for both models.
5. **Dashboard** (`app/dashboard.py`): Streamlit app with charts and an example browser.

## Project structure

```
NLI-Robustness-Benchmark/
├── app/dashboard.py              # Streamlit dashboard
├── data/processed/               # perturbation files, model predictions, final results
├── notebooks/train_scibert_colab.py   # SciBERT training + predictions (run on Colab)
├── src/
│   ├── dataset_loader.py         # load SciNLI, save a stratified sample
│   ├── baseline_tfidf.py         # TF-IDF + LogReg baseline
│   ├── perturbations.py          # rule-based edit generator
│   ├── make_checked.py           # builds the manual-check file
│   ├── auto_valid.py             # pre-fills valid flags, marks risky rows
│   ├── make_changing_sample.py   # samples meaning-changing rows for manual review
│   ├── evaluate_tfidf.py         # quick evaluation on raw perturbations
│   └── evaluate_final.py         # final evaluation on validated rows
├── requirements.txt
└── README.md
```

## Setup and reproduction

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python src/baseline_tfidf.py     # trains TF-IDF baseline, saves models/tfidf_lr.joblib
python src/perturbations.py      # creates data/processed/perturbations_raw.csv
```

SciBERT predictions come from `notebooks/train_scibert_colab.py` (GPU recommended). The repository already includes the resulting `data/processed/scibert_predictions.csv`, so you can run the evaluation and dashboard without retraining:

```bash
python src/evaluate_final.py
streamlit run app/dashboard.py
```

Note: `perturbations.py` overwrites `perturbations_raw.csv`. The manually checked files (`perturbations_checked.csv`, `changing_100.csv`) are kept separate for this reason.

## Limitations

- **Small samples.** Quantity (n=13) and causal (n=18) meaning-changing groups are too small for firm conclusions. Negation (n=23) and qualifier (n=25) are somewhat more reliable.
- **Sensitivity is not accuracy.** SciNLI labels describe the relation between two sentences. After an edit, the new gold label is often ambiguous (an edit may turn `entailment` into `contrasting` or `neutral`), so gold labels were not assigned for meaning-changing edits. A model changing its prediction does not prove it is correct.
- **Rule-based edits are imperfect.** Regex rules can produce awkward sentences. Invalid examples were removed, but the meaning-preserving set was only partly checked by hand.
- **Weak TF-IDF baseline.** TF-IDF has about 50% accuracy on the original pairs, so its higher sensitivity reflects instability rather than understanding.
- **SciBERT was trained on a 40k-pair subset for 2 epochs** to fit a free Colab GPU session.

## Dataset citation

Sadat, M. and Caragea, C. (2022). *SciNLI: A Corpus for Natural Language Inference on Scientific Text.* ACL 2022.
