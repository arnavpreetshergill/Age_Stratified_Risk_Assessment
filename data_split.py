import pandas as pd
import os

from data_pipeline_utils import AGE_GROUPS, assign_age_group_processed, compute_age_z_thresholds
from project_paths import (
    ELDERLY_COHORT_FILE,
    MIDDLE_COHORT_FILE,
    PROCESSED_TRAIN_100K_FILE,
    RAW_DATA_FILE,
    YOUNG_COHORT_FILE,
    ensure_root_artifact_dirs,
)

# --- CONFIGURATION ---
PROCESSED_FILE = PROCESSED_TRAIN_100K_FILE
RAW_FILE = RAW_DATA_FILE

# Output filenames
OUT_YOUNG = YOUNG_COHORT_FILE
OUT_MIDDLE = MIDDLE_COHORT_FILE
OUT_ELDERLY = ELDERLY_COHORT_FILE


def split_by_age_group():
    # 1. Load the Data
    if not os.path.exists(PROCESSED_FILE) or not os.path.exists(RAW_FILE):
        print("Error: Could not find one of the input files.")
        print(f"Looking for: {PROCESSED_FILE} and {RAW_FILE}")
        return

    print("Loading datasets...")
    df_100k = pd.read_csv(PROCESSED_FILE)

    # 2. Calculate Z-Score Thresholds
    z_score_45, z_score_65, raw_mean, raw_std = compute_age_z_thresholds(RAW_FILE)
    
    print(f"Train-Split Age Stats -> Mean: {raw_mean:.2f}, Std: {raw_std:.2f}")
    print(f"Cutoff Thresholds     -> Age 45 (Z={z_score_45:.3f}), Age 65 (Z={z_score_65:.3f})")

    # 3. Perform the Split
    df_100k["age_group"] = assign_age_group_processed(df_100k["age"], z_score_45, z_score_65)
    cohort_frames = {
        age_group: df_100k[df_100k["age_group"] == age_group].drop(columns=["age_group"])
        for age_group in AGE_GROUPS
    }
    df_young = cohort_frames["Young"]
    df_middle = cohort_frames["Middle"]
    df_elderly = cohort_frames["Elderly"]

    # 4. Save to Files
    df_young.to_csv(OUT_YOUNG, index=False)
    df_middle.to_csv(OUT_MIDDLE, index=False)
    df_elderly.to_csv(OUT_ELDERLY, index=False)

    # 5. Print Summary
    print("-" * 30)
    print(f"Successfully split {len(df_100k)} rows:")
    print(f"  - Young (<45):     {len(df_young)} rows saved to '{OUT_YOUNG}'")
    print(f"  - Middle (45-65):  {len(df_middle)} rows saved to '{OUT_MIDDLE}'")
    print(f"  - Elderly (>65):   {len(df_elderly)} rows saved to '{OUT_ELDERLY}'")
    print("-" * 30)

# --- RUN ---
if __name__ == "__main__":
    ensure_root_artifact_dirs()
    split_by_age_group()
