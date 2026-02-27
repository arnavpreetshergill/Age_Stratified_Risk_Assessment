import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_FILENAME = 'processed_train.csv'
OUTPUT_FILENAME = 'processed_train_100k_stratified.csv'
RAW_REFERENCE_FILENAME = 'heart_statlog_cleveland_hungary_final(1).csv'
TARGET_TOTAL = 100000
RANDOM_STATE = 42


def compute_age_z_thresholds(raw_file):
    df_raw = pd.read_csv(raw_file)
    raw_mean = df_raw['age'].mean()
    raw_std = df_raw['age'].std()
    age_z_45 = (45 - raw_mean) / raw_std
    age_z_65 = (65 - raw_mean) / raw_std
    return age_z_45, age_z_65


def get_age_group_scaled(z_score, age_z_45, age_z_65):
    if z_score < age_z_45:
        return 'Young'
    elif age_z_45 <= z_score <= age_z_65:
        return 'Middle'
    else:
        return 'Elderly'


def augment_processed_data(df, target_total=100000, age_z_45=-0.932, age_z_65=1.205, random_state=42):
    print("Identifying column types...")
    rng = np.random.default_rng(random_state)

    # 1. Define Column Groups
    cont_cols = ['age', 'resting_bp_s', 'cholesterol', 'max_heart_rate', 'oldpeak']
    
    # Binary columns (0 or 1)
    bin_cols = ['sex', 'fasting_blood_sugar', 'exercise_angina']
    
    # One-Hot Encoded Groups (Must be handled together to avoid conflicts)
    onehot_groups = {
        'chest_pain': ['chest_pain_type_2', 'chest_pain_type_3', 'chest_pain_type_4'],
        'ecg': ['resting_ecg_1', 'resting_ecg_2'],
        'slope': ['st_slope_2', 'st_slope_3']
    }
    
    expected_cols = set(cont_cols + bin_cols + [col for cols in onehot_groups.values() for col in cols] + ['target'])
    missing_cols = sorted(expected_cols.difference(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns for augmentation: {missing_cols}")

    # 2. Prepare Working DataFrame
    df_work = df.copy()
    df_work['age_grp'] = df_work['age'].apply(lambda z: get_age_group_scaled(z, age_z_45, age_z_65))
    
    # Calculate Needs
    current_len = len(df_work)
    needed = target_total - current_len
    synthetic_samples = []

    if needed <= 0:
        print("No augmentation required. Returning sampled dataset to match target size.")
        sampled = df_work.sample(n=target_total, random_state=random_state).reset_index(drop=True)
        return sampled[df.columns]
    
    # 3. Stratified Loop
    # Calculate proportions based on (Target, Age Group)
    group_counts = df_work.groupby(['target', 'age_grp'], observed=False).size()
    total_obs = len(df_work)
    
    print(f"Starting augmentation. Generating {needed} new samples...")
    
    for target_val in [0, 1]:
        for age_grp in ['Young', 'Middle', 'Elderly']:
            # Filter parents
            group = df_work[(df_work['target'] == target_val) & (df_work['age_grp'] == age_grp)]
            
            # Borrow global parents if local group is empty/tiny
            if len(group) < 2:
                parent_pool = df_work[df_work['target'] == target_val]
            else:
                parent_pool = group
            
            # Calculate how many to make
            obs_count = group_counts.get((target_val, age_grp), 0)
            weight = max(obs_count, 1)
            num_to_create = int(round((weight / total_obs) * needed))
            
            if num_to_create <= 0: continue
            
            # Select Parents
            idx1 = rng.choice(parent_pool.index.to_numpy(), size=num_to_create, replace=True)
            idx2 = rng.choice(parent_pool.index.to_numpy(), size=num_to_create, replace=True)
            alphas = rng.random(num_to_create)
            
            # Generate Synthetic Data
            for i in range(num_to_create):
                p1 = parent_pool.loc[idx1[i]]
                p2 = parent_pool.loc[idx2[i]]
                alpha = alphas[i]
                
                new_row = {'target': target_val}
                
                # A. Continuous Columns: Linear Interpolation
                for col in cont_cols:
                    if col == 'age':
                        # Force age into correct Z-score range
                        if age_grp == 'Young':
                            val = rng.uniform(df['age'].min(), age_z_45)
                        elif age_grp == 'Middle':
                            val = rng.uniform(age_z_45, age_z_65)
                        else:
                            val = rng.uniform(age_z_65, df['age'].max())
                        new_row[col] = val
                    else:
                        # Standard SMOTE interpolation
                        val = p1[col] + alpha * (p2[col] - p1[col])
                        new_row[col] = val

                # B. Binary Columns: Random Choice (0 or 1)
                for col in bin_cols:
                    # Pick from one parent randomly
                    new_row[col] = rng.choice([p1[col], p2[col]])
                
                # C. One-Hot Groups: Inherit WHOLE group from ONE parent
                # (Prevents invalid states like "Chest Pain Type 2 AND Type 3")
                for group_name, cols in onehot_groups.items():
                    donor = p1 if rng.random() > 0.5 else p2
                    for col in cols:
                        new_row[col] = donor[col]
                
                synthetic_samples.append(new_row)
    
    # 4. Final Assembly
    if synthetic_samples:
        df_synthetic = pd.DataFrame(synthetic_samples)
        # Ensure column order matches original
        df_synthetic = df_synthetic[df.columns] 
        augmented_df = pd.concat([df, df_synthetic], ignore_index=True)
    else:
        augmented_df = df
        
    # 5. Fix Exact Count (Trim or Pad)
    if len(augmented_df) > target_total:
        augmented_df = augmented_df.sample(n=target_total, random_state=random_state)
    elif len(augmented_df) < target_total:
        gap = target_total - len(augmented_df)
        extra = augmented_df.sample(n=gap, replace=True, random_state=random_state)
        augmented_df = pd.concat([augmented_df, extra])
        
    return augmented_df.reset_index(drop=True)

# --- EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_FILENAME):
        print(f"Error: '{INPUT_FILENAME}' not found. Please run preprocess.py first.")
    elif not os.path.exists(RAW_REFERENCE_FILENAME):
        print(f"Error: '{RAW_REFERENCE_FILENAME}' not found.")
    else:
        print(f"Loading processed data: {INPUT_FILENAME}")
        df = pd.read_csv(INPUT_FILENAME)
        age_z_45, age_z_65 = compute_age_z_thresholds(RAW_REFERENCE_FILENAME)
        print(f"Using age cutoffs (z-space): 45y={age_z_45:.3f}, 65y={age_z_65:.3f}")

        df_result = augment_processed_data(
            df,
            target_total=TARGET_TOTAL,
            age_z_45=age_z_45,
            age_z_65=age_z_65,
            random_state=RANDOM_STATE,
        )

        df_result.to_csv(OUTPUT_FILENAME, index=False)
        print(f"Success! Saved {len(df_result)} rows to: {OUTPUT_FILENAME}")
        print(df_result.head())
