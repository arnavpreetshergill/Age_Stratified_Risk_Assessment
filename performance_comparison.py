import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split

from preprocess import (
    NUMERIC_FEATURES,
    TARGET_COL,
    build_preprocessor,
    load_and_clean_data,
    transform_with_preprocessor,
)
from performance_visualization_utils import save_visualizations
from project_paths import RAW_DATA_FILE, ensure_root_artifact_dirs
from training_utils import (
    FEATURE_SELECTION_TOP_K,
    select_top_features,
    smote_resample_binary,
    tune_xgboost_with_gridsearch,
)

# --- CONFIGURATION ---
RAW_FILE = RAW_DATA_FILE
TEST_SIZE = 0.2
RANDOM_STATE = 42
AGE_GROUPS = ["Young", "Middle", "Elderly"]
GRIDSEARCH_SCORING = "f1"
MAX_GRIDSEARCH_FOLDS = 5
GRIDSEARCH_PARAM_GRID = {
    "n_estimators": [80, 140],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.15],
    "subsample": [0.8, 1.0],
}


def get_age_cutoffs_from_preprocessor(preprocessor, low_age=45, high_age=65):
    scaler = preprocessor.named_transformers_["num"]
    age_idx = NUMERIC_FEATURES.index("age")
    age_mean = scaler.mean_[age_idx]
    age_std = scaler.scale_[age_idx]
    z45 = (low_age - age_mean) / age_std
    z65 = (high_age - age_mean) / age_std
    return z45, z65, age_mean, age_std


def assign_age_group(age_series, z45, z65):
    conditions = [
        age_series < z45,
        (age_series >= z45) & (age_series <= z65),
        age_series > z65,
    ]
    choices = ["Young", "Middle", "Elderly"]
    return np.select(conditions, choices, default="Unknown")


def compute_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if np.unique(y_true).size > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    return metrics


def fmt(x):
    return "N/A" if pd.isna(x) else f"{x:.4f}"


def fmt_cv(x):
    if x is None:
        return "N/A"
    return "N/A" if pd.isna(x) else f"{x:.4f}"


def format_feature_list(feature_info):
    features = feature_info.get("selected_features", []) if isinstance(feature_info, dict) else []
    return ", ".join(features) if len(features) > 0 else "N/A"


def compare_strategies():
    print("=" * 70)
    print("LEAKAGE-SAFE MODEL COMPARISON: BASELINE vs AGE-STRATIFIED")
    print(f"TOP-{FEATURE_SELECTION_TOP_K} FEATURE SELECTION + TRAIN-ONLY SMOTE")
    print("NO 100K TRAIN AUGMENTATION")
    print("=" * 70)

    if not os.path.exists(RAW_FILE):
        print(f"Error: '{RAW_FILE}' not found.")
        return

    # 1) Clean raw data and split first to avoid leakage.
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

    # 2) Fit preprocessing on train only.
    preprocessor = build_preprocessor()
    train_df = transform_with_preprocessor(X_train_raw, y_train_raw, preprocessor, fit=True)
    test_df = transform_with_preprocessor(X_test_raw, y_test_raw, preprocessor, fit=False)

    z45, z65, age_mean, age_std = get_age_cutoffs_from_preprocessor(preprocessor)
    train_df["age_group"] = assign_age_group(train_df["age"], z45, z65)
    test_df["age_group"] = assign_age_group(test_df["age"], z45, z65)

    print(f"Raw cleaned rows: {len(raw_df)}")
    print(f"Train/Test rows: {len(train_df)} / {len(test_df)}")
    print(
        f"Age thresholds in z-space from train scaler: "
        f"45y={z45:.3f}, 65y={z65:.3f} (mean={age_mean:.3f}, std={age_std:.3f})"
    )

    print(f"Train age groups: {train_df['age_group'].value_counts().to_dict()}")
    print(f"Test age groups:  {test_df['age_group'].value_counts().to_dict()}")
    print("-" * 70)

    feature_cols = [c for c in train_df.columns if c not in [TARGET_COL, "age_group"]]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(int)

    # 3) Baseline model.
    print("1) TUNING + TRAINING BASELINE model...")
    baseline_X_train_selected, baseline_X_test_selected, baseline_feature_info = select_top_features(
        train_df[feature_cols],
        train_df[TARGET_COL],
        X_test,
        top_k=FEATURE_SELECTION_TOP_K,
        random_state=RANDOM_STATE,
    )
    baseline_X_train_final, baseline_y_train_final, baseline_sampling = smote_resample_binary(
        baseline_X_train_selected,
        train_df[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    print(f"   -> Baseline top features: {format_feature_list(baseline_feature_info)}")
    print(
        f"   -> Baseline SMOTE: before {baseline_sampling['before_counts']} "
        f"after {baseline_sampling['after_counts']}"
    )
    baseline_model, baseline_tuning = tune_xgboost_with_gridsearch(
        baseline_X_train_final,
        baseline_y_train_final,
        "Baseline",
        param_grid=GRIDSEARCH_PARAM_GRID,
        scoring=GRIDSEARCH_SCORING,
        max_cv_folds=MAX_GRIDSEARCH_FOLDS,
        random_state=RANDOM_STATE,
    )
    baseline_pred = pd.Series(
        baseline_model.predict(baseline_X_test_selected).astype(int),
        index=test_df.index,
    )
    baseline_prob = pd.Series(
        baseline_model.predict_proba(baseline_X_test_selected)[:, 1],
        index=test_df.index,
    )
    baseline_overall = compute_metrics(y_test, baseline_pred, baseline_prob)

    baseline_group_metrics = {}
    for grp in AGE_GROUPS:
        mask = test_df["age_group"] == grp
        baseline_group_metrics[grp] = compute_metrics(
            y_test[mask],
            baseline_pred[mask],
            baseline_prob[mask],
        )

    # 4) Stratified specialist models.
    print("2) TUNING + TRAINING STRATIFIED specialists...")
    strat_pred = pd.Series(index=test_df.index, dtype=float)
    strat_prob = pd.Series(index=test_df.index, dtype=float)
    strat_group_metrics = {}
    strat_tuning = {}
    strat_feature_selection = {}
    strat_sampling = {}

    for grp in AGE_GROUPS:
        train_grp = train_df[train_df["age_group"] == grp]
        test_grp = test_df[test_df["age_group"] == grp]

        if train_grp.empty or test_grp.empty or train_grp[TARGET_COL].nunique() < 2:
            print(f"   -> {grp}: skipped (insufficient train/test class coverage)")
            strat_group_metrics[grp] = {
                "accuracy": np.nan,
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "tp": 0,
            }
            strat_tuning[grp] = {
                "status": "skipped",
                "reason": "insufficient_train_test_class_coverage",
                "cv_folds": 0,
                "param_grid_size": len(list(ParameterGrid(GRIDSEARCH_PARAM_GRID))),
                "best_params": None,
                "best_cv_score_f1": None,
            }
            strat_feature_selection[grp] = None
            strat_sampling[grp] = None
            continue

        X_train_grp_selected, X_test_grp_selected, feature_info = select_top_features(
            train_grp[feature_cols],
            train_grp[TARGET_COL],
            test_grp[feature_cols],
            top_k=FEATURE_SELECTION_TOP_K,
            random_state=RANDOM_STATE,
        )
        X_train_grp_final, y_train_grp_final, sampling_info = smote_resample_binary(
            X_train_grp_selected,
            train_grp[TARGET_COL],
            random_state=RANDOM_STATE,
        )
        specialist, tuning = tune_xgboost_with_gridsearch(
            X_train_grp_final,
            y_train_grp_final,
            grp,
            param_grid=GRIDSEARCH_PARAM_GRID,
            scoring=GRIDSEARCH_SCORING,
            max_cv_folds=MAX_GRIDSEARCH_FOLDS,
            random_state=RANDOM_STATE,
        )
        grp_pred = specialist.predict(X_test_grp_selected).astype(int)
        grp_prob = specialist.predict_proba(X_test_grp_selected)[:, 1]

        strat_pred.loc[test_grp.index] = grp_pred
        strat_prob.loc[test_grp.index] = grp_prob
        strat_group_metrics[grp] = compute_metrics(
            test_grp[TARGET_COL].astype(int),
            grp_pred,
            grp_prob,
        )
        strat_tuning[grp] = tuning
        strat_feature_selection[grp] = feature_info
        strat_sampling[grp] = sampling_info
        print(f"   -> {grp}: Accuracy {strat_group_metrics[grp]['accuracy']:.4f}")
        print(f"      top features: {format_feature_list(feature_info)}")
        print(f"      SMOTE: before {sampling_info['before_counts']} after {sampling_info['after_counts']}")

    # Fallback if any rows were not scored by specialists.
    missing_idx = strat_pred[strat_pred.isna()].index
    if len(missing_idx) > 0:
        print(f"   -> Warning: {len(missing_idx)} test rows missing specialist predictions, using baseline.")
        strat_pred.loc[missing_idx] = baseline_pred.loc[missing_idx]
        strat_prob.loc[missing_idx] = baseline_prob.loc[missing_idx]

    strat_pred = strat_pred.astype(int)
    strat_overall = compute_metrics(y_test, strat_pred, strat_prob)

    # 5) Reporting.
    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print(f"{'Metric':<12} | {'Baseline':<10} | {'Stratified':<10} | {'Winner':<10}")
    print("-" * 70)
    for metric in ["accuracy", "roc_auc", "pr_auc", "recall", "f1"]:
        b = baseline_overall[metric]
        s = strat_overall[metric]
        if pd.isna(b) or pd.isna(s):
            winner = "N/A"
        elif s > b:
            winner = "Stratified"
        elif b > s:
            winner = "Baseline"
        else:
            winner = "Tie"
        print(f"{metric:<12} | {fmt(b):<10} | {fmt(s):<10} | {winner:<10}")

    print("-" * 70)
    print(
        f"Baseline CM [TN FP FN TP]: "
        f"[{baseline_overall['tn']} {baseline_overall['fp']} {baseline_overall['fn']} {baseline_overall['tp']}]"
    )
    print(
        f"Stratified CM [TN FP FN TP]: "
        f"[{strat_overall['tn']} {strat_overall['fp']} {strat_overall['fn']} {strat_overall['tp']}]"
    )

    print("\nPER-GROUP ACCURACY")
    print(f"{'Age Group':<10} | {'Baseline':<10} | {'Stratified':<10}")
    print("-" * 70)
    for grp in AGE_GROUPS:
        print(
            f"{grp:<10} | "
            f"{fmt(baseline_group_metrics[grp]['accuracy']):<10} | "
            f"{fmt(strat_group_metrics[grp]['accuracy']):<10}"
        )

    print("\nTUNING SUMMARY")
    print("-" * 70)
    print(
        f"Baseline best params: {baseline_tuning['best_params']} "
        f"| best CV f1: {fmt_cv(baseline_tuning['best_cv_score_f1'])}"
    )
    for grp in AGE_GROUPS:
        tuning = strat_tuning.get(grp, {})
        print(
            f"{grp:<10} params: {tuning.get('best_params')} "
            f"| best CV f1: {fmt_cv(tuning.get('best_cv_score_f1'))}"
        )

    print("\nFEATURE SELECTION + SMOTE SUMMARY")
    print("-" * 70)
    print(f"Baseline top features: {format_feature_list(baseline_feature_info)}")
    print(
        f"Baseline SMOTE counts: before {baseline_sampling['before_counts']} "
        f"after {baseline_sampling['after_counts']}"
    )
    for grp in AGE_GROUPS:
        print(f"{grp:<10} features: {format_feature_list(strat_feature_selection.get(grp))}")
        sampling_info = strat_sampling.get(grp)
        if sampling_info is None:
            print(f"{grp:<10} SMOTE: N/A")
            continue
        print(
            f"{grp:<10} SMOTE: before {sampling_info['before_counts']} "
            f"after {sampling_info['after_counts']}"
        )

    visualization_paths = save_visualizations(
        baseline_overall,
        strat_overall,
        baseline_group_metrics,
        strat_group_metrics,
        baseline_sampling,
        strat_sampling,
        baseline_feature_info,
        strat_feature_selection,
    )
    print("\nVISUALIZATION OUTPUTS")
    print("-" * 70)
    for visualization_path in visualization_paths:
        print(f"- {visualization_path}")


if __name__ == "__main__":
    ensure_root_artifact_dirs()
    compare_strategies()
