import os
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from data_pipeline_utils import TARGET_COL, assign_age_group_raw, prepare_train_test_data
from holdout_utils import fit_final_xgboost_pipeline, fit_xgboost_pipeline
from project_paths import RAW_DATA_FILE, SHAP_VISUALIZATIONS_DIR, ensure_root_artifact_dirs
from training_utils import FEATURE_SELECTION_TOP_K

# --- CONFIGURATION ---
RAW_FILE = RAW_DATA_FILE
COHORTS = [
    {"key": "Young", "name": "Young Adults (<45)", "output": "Young"},
    {"key": "Middle", "name": "Middle-Aged (45-65)", "output": "Middle-Aged"},
    {"key": "Elderly", "name": "Elderly (>65)", "output": "Elderly"},
]
RANDOM_STATE = 42


def compute_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        return {
            "accuracy": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
        }

    accuracy = float(np.mean(y_true == y_pred))
    recall = 0.0
    f1 = 0.0
    roc_auc = np.nan

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    if tp + fn > 0:
        recall = tp / (tp + fn)
    if (2 * tp + fp + fn) > 0:
        f1 = (2 * tp) / (2 * tp + fp + fn)
    if np.unique(y_true).size > 1:
        from sklearn.metrics import roc_auc_score

        roc_auc = float(roc_auc_score(y_true, y_prob))

    return {
        "accuracy": accuracy,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def format_feature_list(feature_info):
    features = feature_info.get("selected_features", []) if isinstance(feature_info, dict) else []
    return ", ".join(features) if len(features) > 0 else "N/A"


def wilson_accuracy_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p_hat = correct / total
    denom = 1 + (z * z / total)
    center = (p_hat + (z * z / (2 * total))) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) / total) + (z * z / (4 * total * total))) / denom
    return float(center - margin), float(center + margin)


def analyze_group(
    train_group_df: pd.DataFrame,
    test_group_df: pd.DataFrame,
    full_group_df: pd.DataFrame,
    group_name: str,
    output_name: str,
):
    print("=" * 60)
    print(f"STARTING ANALYSIS: {group_name}")
    print(f"TOP-{FEATURE_SELECTION_TOP_K} FEATURE SELECTION + TRAIN-ONLY SMOTE")
    print("EVALUATION MODE: FIXED TRAIN/TEST HOLDOUT")
    print("=" * 60)

    train_feature_cols = [column for column in train_group_df.columns if column != TARGET_COL]
    test_feature_cols = [column for column in test_group_df.columns if column != TARGET_COL]
    print(
        f"Train rows: {len(train_group_df)} | Test rows: {len(test_group_df)} "
        f"| Train class counts: {train_group_df[TARGET_COL].astype(int).value_counts().sort_index().to_dict()}"
    )

    if len(test_group_df) == 0:
        print(f"Skipping {group_name}: empty test cohort.")
        return

    if len(train_group_df) == 0 or train_group_df[TARGET_COL].nunique() < 2:
        print(f"Skipping {group_name}: insufficient class diversity in training cohort.")
        return

    holdout_result = fit_xgboost_pipeline(
        train_group_df[train_feature_cols].reset_index(drop=True),
        train_group_df[TARGET_COL].astype(int).reset_index(drop=True),
        test_group_df[test_feature_cols].reset_index(drop=True),
        label=f"{group_name} Holdout",
        top_k=FEATURE_SELECTION_TOP_K,
        use_smote=True,
        random_state=RANDOM_STATE,
    )

    holdout_metrics = compute_metrics(
        test_group_df[TARGET_COL].astype(int).to_numpy(),
        holdout_result["pred"],
        holdout_result["prob"],
    )
    correct = int((holdout_result["pred"] == test_group_df[TARGET_COL].astype(int).to_numpy()).sum())
    ci_low, ci_high = wilson_accuracy_interval(correct, len(test_group_df))
    print(
        f"Holdout accuracy for {group_name}: {holdout_metrics['accuracy']:.2%} "
        f"(95% CI {ci_low:.2%} to {ci_high:.2%})"
    )
    print(
        f"Holdout ROC-AUC: {holdout_metrics['roc_auc']:.4f}"
        if not pd.isna(holdout_metrics["roc_auc"])
        else "Holdout ROC-AUC: N/A"
    )
    print(f"Params: {holdout_result['model_info']['params']}")
    print(f"Feature selection: {format_feature_list(holdout_result['feature_selection'])}")
    print(
        f"SMOTE counts: before {holdout_result['sampling']['before_counts']} "
        f"after {holdout_result['sampling']['after_counts']}"
    )
    print("-" * 30)

    full_feature_cols = [column for column in full_group_df.columns if column != TARGET_COL]
    final_fit = fit_final_xgboost_pipeline(
        full_group_df[full_feature_cols],
        full_group_df[TARGET_COL].astype(int),
        label=f"{group_name} Final Explanation Model",
        top_k=FEATURE_SELECTION_TOP_K,
        use_smote=True,
        random_state=RANDOM_STATE,
    )
    X_explain = final_fit["X_eval_selected"]

    print(f"Generating SHAP plots for {group_name} using the full cohort fit...")
    explainer = shap.Explainer(final_fit["model"], X_explain)
    shap_values = explainer(X_explain)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_explain, show=False)
    plt.title(f"SHAP Summary: {group_name}")
    plot_filename = SHAP_VISUALIZATIONS_DIR / f"shap_summary_{output_name}.png"
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_filename}")

    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title(f"Feature Importance: {group_name}")
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
        split_data = prepare_train_test_data(RAW_FILE, random_state=RANDOM_STATE)
        X_train_raw = split_data["X_train_raw"]
        X_test_raw = split_data["X_test_raw"]
        train_df = split_data["train_df"].copy()
        test_df = split_data["test_df"].copy()

        train_df["age_group"] = assign_age_group_raw(X_train_raw["age"])
        test_df["age_group"] = assign_age_group_raw(X_test_raw["age"])

        full_df = pd.concat([train_df, test_df], ignore_index=True)
        print(
            f"Using the fixed train/test split from {RAW_FILE}: "
            f"train={len(train_df)} rows, test={len(test_df)} rows"
        )
        print(f"Train age group counts: {train_df['age_group'].value_counts().to_dict()}")
        print(f"Test age group counts: {test_df['age_group'].value_counts().to_dict()}")

        for cohort in COHORTS:
            group_key = cohort["key"]
            analyze_group(
                train_df[train_df["age_group"] == group_key].drop(columns=["age_group"]).copy(),
                test_df[test_df["age_group"] == group_key].drop(columns=["age_group"]).copy(),
                full_df[full_df["age_group"] == group_key].drop(columns=["age_group"]).copy(),
                cohort["name"],
                cohort["output"],
            )

        print("Done! Check your folder for the PNG images.")
