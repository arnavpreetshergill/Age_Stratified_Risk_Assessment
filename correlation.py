import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_pipeline_utils import TARGET_COL
from project_paths import CORRELATION_HEATMAP_FILE, PROCESSED_TRAIN_FILE, ensure_root_artifact_dirs

DATA_FILE = PROCESSED_TRAIN_FILE
OUTPUT_FILE = CORRELATION_HEATMAP_FILE


def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found. Please run preprocess.py first.")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"Loaded dataset: {DATA_FILE} ({df.shape[0]} rows, {df.shape[1]} columns)")

    numerical_df = df.select_dtypes(include=["float64", "int64"])
    if TARGET_COL in numerical_df.columns:
        numerical_df = numerical_df.drop(columns=[TARGET_COL])
    if numerical_df.empty:
        print("Error: no numeric columns available for correlation analysis.")
        return

    corr_matrix = numerical_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        mask=mask,
    )
    plt.title("Processed Training Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    ensure_root_artifact_dirs()
    main()
