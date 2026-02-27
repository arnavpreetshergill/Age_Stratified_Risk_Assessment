import pandas as pd
import os

# --- CONFIGURATION ---
PROCESSED_FILE = 'processed_train_100k_stratified.csv'
RAW_FILE = 'heart_statlog_cleveland_hungary_final(1).csv'

# Output filenames
OUT_YOUNG = 'heart_data_young.csv'
OUT_MIDDLE = 'heart_data_middle.csv'
OUT_ELDERLY = 'heart_data_elderly.csv'

def split_by_age_group():
    # 1. Load the Data
    if not os.path.exists(PROCESSED_FILE) or not os.path.exists(RAW_FILE):
        print("Error: Could not find one of the input files.")
        print(f"Looking for: {PROCESSED_FILE} and {RAW_FILE}")
        return

    print("Loading datasets...")
    df_100k = pd.read_csv(PROCESSED_FILE)
    df_raw = pd.read_csv(RAW_FILE)

    # 2. Calculate Z-Score Thresholds
    # We need to know what "Age 45" and "Age 65" look like in Z-scores
    raw_mean = df_raw['age'].mean()
    raw_std = df_raw['age'].std()
    
    z_score_45 = (45 - raw_mean) / raw_std
    z_score_65 = (65 - raw_mean) / raw_std
    
    print(f"Original Age Stats -> Mean: {raw_mean:.2f}, Std: {raw_std:.2f}")
    print(f"Cutoff Thresholds  -> Age 45 (Z={z_score_45:.3f}), Age 65 (Z={z_score_65:.3f})")

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
    split_by_age_group()
