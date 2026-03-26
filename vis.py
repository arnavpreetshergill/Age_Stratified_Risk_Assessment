import matplotlib
matplotlib.use("Agg")
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_pipeline_utils import AGE_GROUPS, assign_age_group_processed, compute_age_z_thresholds
from project_paths import (
    AGE_GROUP_DISTRIBUTION_FILE,
    PROCESSED_TRAIN_100K_FILE,
    RAW_DATA_FILE,
    ensure_root_artifact_dirs,
)

DATA_FILE = PROCESSED_TRAIN_100K_FILE
RAW_FILE = RAW_DATA_FILE
OUTPUT_FILE = AGE_GROUP_DISTRIBUTION_FILE


def main():
    if not os.path.exists(DATA_FILE) or not os.path.exists(RAW_FILE):
        print(f"Error: required files not found. Expected {DATA_FILE} and {RAW_FILE}.")
        return

    df = pd.read_csv(DATA_FILE)
    z45, z65, mean_age, std_age = compute_age_z_thresholds(RAW_FILE)
    print(
        f"Using train-split z-cutoffs: 45y={z45:.3f}, 65y={z65:.3f} "
        f"(mean={mean_age:.3f}, std={std_age:.3f})"
    )

    df["age_group"] = assign_age_group_processed(df["age"], z45, z65)
    sns.countplot(x="age_group", hue="target", data=df, order=AGE_GROUPS, hue_order=[0, 1])
    plt.title("Generated Training Distribution Across Age Groups")
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=200)
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    ensure_root_artifact_dirs()
    main()
