import pandas as pd
import os
from sklearn.model_selection import train_test_split

from preprocess import NUMERIC_FEATURES, TARGET_COL, build_preprocessor, load_and_clean_data
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


def compute_age_z_thresholds(raw_file, test_size=0.2, random_state=42):
    df_clean = load_and_clean_data(raw_file)
    X_raw = df_clean.drop(columns=[TARGET_COL])
    y_raw = df_clean[TARGET_COL].astype(int)

    X_train_raw, _, _, _ = train_test_split(
        X_raw,
        y_raw,
        test_size=test_size,
        random_state=random_state,
        stratify=y_raw,
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train_raw)
    scaler = preprocessor.named_transformers_["num"]
    age_idx = NUMERIC_FEATURES.index("age")
    raw_mean = float(scaler.mean_[age_idx])
    raw_std = float(scaler.scale_[age_idx])
    z_score_45 = (45 - raw_mean) / raw_std
    z_score_65 = (65 - raw_mean) / raw_std
    return z_score_45, z_score_65, raw_mean, raw_std

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
    # Young: Age < 45
    df_young = df_100k[df_100k['age'] < z_score_45]
    
    # Middle: 45 <= Age <= 65
    df_middle = df_100k[(df_100k['age'] >= z_score_45) & (df_100k['age'] <= z_score_65)]
    
    # Elderly: Age > 65
    df_elderly = df_100k[df_100k['age'] > z_score_65]

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
