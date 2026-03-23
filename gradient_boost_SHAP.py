import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from preprocess import (
    NUMERIC_FEATURES,
    TARGET_COL,
    build_preprocessor,
    load_and_clean_data,
    transform_with_preprocessor,
)
from project_paths import RAW_DATA_FILE, SHAP_VISUALIZATIONS_DIR, ensure_root_artifact_dirs
from training_utils import (
    FEATURE_SELECTION_TOP_K,
    select_top_features,
    smote_resample_binary,
    tune_xgboost_with_gridsearch,
)

# --- CONFIGURATION ---
RAW_FILE = RAW_DATA_FILE
TEST_SIZE = 0.2
COHORTS = [
    {"key": "Young", "name": "Young Adults (<45)", "output": "Young"},
    {"key": "Middle", "name": "Middle-Aged (45-65)", "output": "Middle-Aged"},
    {"key": "Elderly", "name": "Elderly (>65)", "output": "Elderly"},
]
RANDOM_STATE = 42
GRIDSEARCH_SCORING = "f1"
MAX_GRIDSEARCH_FOLDS = 5
GRIDSEARCH_PARAM_GRID = {
    "n_estimators": [80, 140],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.15],
    "subsample": [0.8, 1.0],
}


def format_feature_list(feature_info):
    features = feature_info.get("selected_features", []) if isinstance(feature_info, dict) else []
    return ", ".join(features) if len(features) > 0 else "N/A"


def get_age_cutoffs_from_preprocessor(preprocessor, low_age=45, high_age=65):
    scaler = preprocessor.named_transformers_["num"]
    age_idx = NUMERIC_FEATURES.index("age")
    age_mean = float(scaler.mean_[age_idx])
    age_std = float(scaler.scale_[age_idx])
    z45 = (low_age - age_mean) / age_std
    z65 = (high_age - age_mean) / age_std
    return z45, z65, age_mean, age_std


def assign_age_group(age_series, z45, z65):
    conditions = [
        age_series < z45,
        (age_series >= z45) & (age_series <= z65),
        age_series > z65,
    ]
    return np.select(conditions, ["Young", "Middle", "Elderly"], default="Unknown")


def analyze_group(train_df, test_df, group_name, output_name):
    print("=" * 60)
    print(f"STARTING ANALYSIS: {group_name}")
    print(f"TOP-{FEATURE_SELECTION_TOP_K} FEATURE SELECTION + TRAIN-ONLY SMOTE")
    print("=" * 60)
    print(f"Train/Test rows: {len(train_df)} / {len(test_df)}")

    if len(train_df) == 0 or len(test_df) == 0 or train_df[TARGET_COL].nunique() < 2:
        print(f"Skipping {group_name}: insufficient cohort train/test coverage.")
        return

    X_train = train_df.drop(columns=[TARGET_COL, "age_group"])
    y_train = train_df[TARGET_COL].astype(int)
    X_test = test_df.drop(columns=[TARGET_COL, "age_group"])
    y_test = test_df[TARGET_COL].astype(int)

    X_train_selected, X_test_selected, feature_info = select_top_features(
        X_train,
        y_train,
        X_test,
        top_k=FEATURE_SELECTION_TOP_K,
        random_state=RANDOM_STATE,
    )
    X_train_final, y_train_final, sampling_info = smote_resample_binary(
        X_train_selected,
        y_train,
        random_state=RANDOM_STATE,
    )
    print(f"Selected features: {format_feature_list(feature_info)}")
    print(
        f"SMOTE class counts: before {sampling_info['before_counts']} "
        f"after {sampling_info['after_counts']}"
    )
    model, _ = tune_xgboost_with_gridsearch(
        X_train_final,
        y_train_final,
        group_name,
        param_grid=GRIDSEARCH_PARAM_GRID,
        scoring=GRIDSEARCH_SCORING,
        max_cv_folds=MAX_GRIDSEARCH_FOLDS,
        random_state=RANDOM_STATE,
    )

    y_pred = model.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy for {group_name}: {acc:.2%}")
    print("-" * 30)

    print(f"Generating SHAP plots for {group_name}...")
    explainer = shap.Explainer(model, X_test_selected)
    shap_values = explainer(X_test_selected)

    plt.figure(figsize=(10, 6))
    plt.title(f"SHAP Summary: {group_name}")
    shap.summary_plot(shap_values, X_test_selected, show=False)
    plot_filename = SHAP_VISUALIZATIONS_DIR / f"shap_summary_{output_name}.png"
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_filename}")

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance: {group_name}")
    shap.plots.bar(shap_values, show=False)
    bar_filename = SHAP_VISUALIZATIONS_DIR / f"shap_importance_{output_name}.png"
    plt.savefig(bar_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved: {bar_filename}")
    print()


if __name__ == "__main__":
    ensure_root_artifact_dirs()
    if not os.path.exists(RAW_FILE):
        print(f"Error: '{RAW_FILE}' not found.")
    else:
        raw_df = load_and_clean_data(RAW_FILE)
        X_raw = raw_df.drop(columns=[TARGET_COL])
        y_raw = raw_df[TARGET_COL].astype(int)

        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_raw,
            y_raw,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_raw,
        )

        preprocessor = build_preprocessor()
        train_df = transform_with_preprocessor(X_train_raw, y_train_raw, preprocessor, fit=True)
        test_df = transform_with_preprocessor(X_test_raw, y_test_raw, preprocessor, fit=False)

        z45, z65, age_mean, age_std = get_age_cutoffs_from_preprocessor(preprocessor)
        train_df["age_group"] = assign_age_group(train_df["age"], z45, z65)
        test_df["age_group"] = assign_age_group(test_df["age"], z45, z65)

        print(
            f"Using clean train/test split from {RAW_FILE}: "
            f"{len(train_df)} train rows, {len(test_df)} test rows"
        )
        print(
            f"Age thresholds in z-space from train scaler: "
            f"45y={z45:.3f}, 65y={z65:.3f} (mean={age_mean:.3f}, std={age_std:.3f})"
        )

        for cohort in COHORTS:
            analyze_group(
                train_df[train_df["age_group"] == cohort["key"]].copy(),
                test_df[test_df["age_group"] == cohort["key"]].copy(),
                cohort["name"],
                cohort["output"],
            )

        print("Done! Check your folder for the PNG images.")
