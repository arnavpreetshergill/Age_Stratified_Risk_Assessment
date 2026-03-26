import pandas as pd
import numpy as np
import os

from data_pipeline_utils import (
    NUMERIC_FEATURES,
    TARGET_COL,
    assert_frames_match,
    assign_age_group_processed,
    get_processed_age_cutoffs,
    prepare_train_test_data,
)
from project_paths import (
    PROCESSED_TRAIN_100K_FILE,
    PROCESSED_TRAIN_FILE,
    RAW_DATA_FILE,
    ensure_root_artifact_dirs,
)
from training_utils import BINARY_FEATURES, ONE_HOT_GROUPS

# --- CONFIGURATION ---
INPUT_FILENAME = PROCESSED_TRAIN_FILE
OUTPUT_FILENAME = PROCESSED_TRAIN_100K_FILE
RAW_REFERENCE_FILENAME = RAW_DATA_FILE
TARGET_TOTAL = 100000
RANDOM_STATE = 42


def _allocate_counts_exact(group_keys, weights, target_total):
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Allocation weights must sum to a positive value.")

    raw_allocations = {
        group_key: (target_total * weights[group_key]) / total_weight for group_key in group_keys
    }
    base_allocations = {group_key: int(np.floor(raw_allocations[group_key])) for group_key in group_keys}
    remainder = target_total - sum(base_allocations.values())
    if remainder > 0:
        ranked_remainders = sorted(
            group_keys,
            key=lambda group_key: raw_allocations[group_key] - base_allocations[group_key],
            reverse=True,
        )
        for group_key in ranked_remainders[:remainder]:
            base_allocations[group_key] += 1
    return base_allocations


def augment_processed_data(df, target_total=100000, age_z_45=None, age_z_65=None, random_state=42):
    print("Identifying column types...")
    rng = np.random.default_rng(random_state)
    if target_total < 0:
        raise ValueError("target_total must be non-negative.")
    if age_z_45 is None or age_z_65 is None:
        raise ValueError("age_z_45 and age_z_65 must be provided from the train-fitted preprocessor.")

    # 1. Define Column Groups
    cont_cols = list(NUMERIC_FEATURES)
    bin_cols = sorted(BINARY_FEATURES)
    onehot_groups = [list(group) for group in ONE_HOT_GROUPS]

    expected_cols = set(
        cont_cols + bin_cols + [col for cols in onehot_groups for col in cols] + [TARGET_COL]
    )
    missing_cols = sorted(expected_cols.difference(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns for augmentation: {missing_cols}")

    # 2. Prepare Working DataFrame
    df_work = df.copy()
    df_work["age_grp"] = assign_age_group_processed(df_work["age"], age_z_45, age_z_65)

    # Calculate Needs
    current_len = len(df_work)
    needed = target_total - current_len
    synthetic_samples = []

    if needed <= 0:
        if needed == 0:
            print("No augmentation required. Returning the input dataset unchanged.")
            return df.reset_index(drop=True)
        if target_total == 0:
            print("Target size is 0. Returning an empty dataset with the original schema.")
            return df.iloc[0:0].copy()

        print("Target size is smaller than the input dataset. Downsampling proportionally by target and age group.")
        group_keys = [(target_val, age_grp) for target_val in [0, 1] for age_grp in ["Young", "Middle", "Elderly"]]
        group_counts = df_work.groupby([TARGET_COL, "age_grp"], observed=False).size()
        downsample_weights = {
            group_key: int(group_counts.get(group_key, 0)) for group_key in group_keys
        }
        allocated_counts = _allocate_counts_exact(group_keys, downsample_weights, target_total)
        sampled_groups = []
        for target_val, age_grp in group_keys:
            group = df_work[(df_work[TARGET_COL] == target_val) & (df_work["age_grp"] == age_grp)]
            sample_count = allocated_counts[(target_val, age_grp)]
            if sample_count <= 0:
                continue
            sampled_groups.append(group.sample(n=sample_count, replace=False, random_state=random_state))
        sampled = pd.concat(sampled_groups, ignore_index=True)
        sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        return sampled[df.columns]
    
    group_keys = [(target_val, age_grp) for target_val in [0, 1] for age_grp in ["Young", "Middle", "Elderly"]]
    group_counts = df_work.groupby([TARGET_COL, "age_grp"], observed=False).size()
    weights = {group_key: int(group_counts.get(group_key, 0)) for group_key in group_keys}
    base_allocations = _allocate_counts_exact(group_keys, weights, needed)

    print(f"Starting augmentation. Generating {needed} new samples...")

    for target_val, age_grp in group_keys:
        group = df_work[(df_work[TARGET_COL] == target_val) & (df_work["age_grp"] == age_grp)]

        # Borrow target-specific parents if the cohort is empty or too small for interpolation.
        if len(group) < 2:
            parent_pool = df_work[df_work[TARGET_COL] == target_val]
        else:
            parent_pool = group

        num_to_create = base_allocations[(target_val, age_grp)]
        if num_to_create <= 0:
            continue
        if parent_pool.empty:
            raise ValueError(
                f"Cannot synthesize target={target_val}, age_group={age_grp} because no parent rows exist."
            )

        idx1 = rng.choice(parent_pool.index.to_numpy(), size=num_to_create, replace=True)
        idx2 = rng.choice(parent_pool.index.to_numpy(), size=num_to_create, replace=True)
        alphas = rng.random(num_to_create)

        for i in range(num_to_create):
            p1 = parent_pool.loc[idx1[i]]
            p2 = parent_pool.loc[idx2[i]]
            alpha = alphas[i]

            new_row = {TARGET_COL: target_val}

            for col in cont_cols:
                if col == "age":
                    if age_grp == "Young":
                        val = rng.uniform(df["age"].min(), age_z_45)
                    elif age_grp == "Middle":
                        val = rng.uniform(age_z_45, age_z_65)
                    else:
                        val = rng.uniform(age_z_65, df["age"].max())
                    new_row[col] = val
                else:
                    new_row[col] = p1[col] + alpha * (p2[col] - p1[col])

            for col in bin_cols:
                new_row[col] = rng.choice([p1[col], p2[col]])

            for cols in onehot_groups:
                donor = p1 if rng.random() > 0.5 else p2
                for col in cols:
                    new_row[col] = donor[col]

            synthetic_samples.append(new_row)

    # 4. Final Assembly
    if synthetic_samples:
        df_synthetic = pd.DataFrame(synthetic_samples)
        df_synthetic = df_synthetic[df.columns]
        augmented_df = pd.concat([df, df_synthetic], ignore_index=True)
    else:
        augmented_df = df

    if len(augmented_df) != target_total:
        raise ValueError(
            f"Augmentation produced {len(augmented_df)} rows, expected exactly {target_total}."
        )

    return augmented_df.reset_index(drop=True)

# --- EXECUTION ---
if __name__ == "__main__":
    ensure_root_artifact_dirs()
    if not os.path.exists(INPUT_FILENAME):
        print(f"Error: '{INPUT_FILENAME}' not found. Please run preprocess.py first.")
    elif not os.path.exists(RAW_REFERENCE_FILENAME):
        print(f"Error: '{RAW_REFERENCE_FILENAME}' not found.")
    else:
        print(f"Loading processed data: {INPUT_FILENAME}")
        df = pd.read_csv(INPUT_FILENAME)
        split_data = prepare_train_test_data(
            RAW_REFERENCE_FILENAME,
            random_state=RANDOM_STATE,
        )
        assert_frames_match(split_data["train_df"], df, "processed_train.csv")
        age_z_45, age_z_65 = get_processed_age_cutoffs(split_data["preprocessor"])
        scaler = split_data["preprocessor"].named_transformers_["num"]
        raw_mean = float(scaler.mean_[NUMERIC_FEATURES.index("age")])
        raw_std = float(scaler.scale_[NUMERIC_FEATURES.index("age")])
        print(
            f"Using train-split age cutoffs (z-space): 45y={age_z_45:.3f}, 65y={age_z_65:.3f} "
            f"(mean={raw_mean:.3f}, std={raw_std:.3f})"
        )

        df_result = augment_processed_data(
            df,
            target_total=TARGET_TOTAL,
            age_z_45=age_z_45,
            age_z_65=age_z_65,
            random_state=RANDOM_STATE,
        )

        df_result.to_csv(OUTPUT_FILENAME, index=False)
        print(f"Success! Saved {len(df_result)} rows to: {OUTPUT_FILENAME}")
