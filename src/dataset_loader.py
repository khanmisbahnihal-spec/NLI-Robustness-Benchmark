
from datasets import load_dataset
import pandas as pd
from pathlib import Path


def main():
    print("Loading SciNLI dataset...")

    dataset = load_dataset("tasksource/scinli")

    print("\nAvailable splits:")
    for split_name, split_data in dataset.items():
        print(f"{split_name}: {len(split_data)} rows")

    # Convert the training split to a DataFrame
    train_df = dataset["train"].to_pandas()

    print("\nColumns:")
    print(train_df.columns.tolist())

    print("\nFirst 5 rows:")
    print(train_df.head())

    print("\nLabel distribution:")
    print(train_df["label"].value_counts(dropna=False))

    # Save a small sample for initial experiments
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = (
        train_df.sample(frac=1, random_state=42)
        .groupby("label")
        .head(250)
        .reset_index(drop=True)
    )
    sample.to_csv(
        output_dir / "scinli_sample.csv",
        index=False
    )

    print("\nSaved 1000 training examples.")
    print("Location: data/processed/scinli_sample.csv")


if __name__ == "__main__":
    main()
