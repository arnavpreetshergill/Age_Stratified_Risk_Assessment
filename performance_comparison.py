from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

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

from data_pipeline_utils import AGE_GROUPS, TARGET_COL, assign_age_group_raw, prepare_train_test_data
from holdout_utils import default_model_info, fit_model_pipeline
from performance_visualization_utils import save_cross_model_visualizations, save_model_visualizations
from project_paths import PERFORMANCE_VISUALIZATIONS_DIR, RAW_DATA_FILE, ensure_root_artifact_dirs
from training_utils import FEATURE_SELECTION_TOP_K, get_available_model_keys, get_model_display_name, get_unavailable_models

# --- CONFIGURATION ---
RAW_FILE = RAW_DATA_FILE
RANDOM_STATE = 42
RESULTS_JSON_PATH = PERFORMANCE_VISUALIZATIONS_DIR / "comparison_results.json"
SUMMARY_CSV_PATH = PERFORMANCE_VISUALIZATIONS_DIR / "cross_model_summary.csv"


def empty_metrics() -> Dict[str, Any]:
    return {
        "accuracy": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "tp": 0,
    }


def compute_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        return empty_metrics()

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


def format_feature_list(feature_info):
    features = feature_info.get("selected_features", []) if isinstance(feature_info, dict) else []
    return ", ".join(features) if len(features) > 0 else "N/A"


def skipped_sampling_info(y_train: pd.Series, reason: str) -> Dict[str, Any]:
    counts = y_train.astype(int).value_counts().sort_index()
    count_map = {0: 0, 1: 0}
    for label, count in counts.items():
        count_map[int(label)] = int(count)
    return {
        "applied": False,
        "reason": reason,
        "before_counts": count_map,
        "after_counts": count_map.copy(),
        "generated_samples": 0,
        "k_neighbors_used": 0,
    }


def winner_for_metric(baseline_value: float, specialist_value: float) -> str:
    if pd.isna(baseline_value) or pd.isna(specialist_value):
        return "N/A"
    if specialist_value > baseline_value:
        return "Age-Specialist"
    if baseline_value > specialist_value:
        return "Baseline"
    return "Tie"


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value) if isinstance(value, (float, np.floating)) else False:
        return None
    return value


def build_cross_model_summary(results_by_model: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_key, model_result in results_by_model.items():
        row = {
            "model_key": model_key,
            "model_name": model_result["model_name"],
        }
        for strategy_key, prefix in [("baseline", "baseline"), ("age_specialist", "age_specialist")]:
            overall = model_result[strategy_key]["overall"]
            row[f"{prefix}_accuracy"] = overall["accuracy"]
            row[f"{prefix}_roc_auc"] = overall["roc_auc"]
            row[f"{prefix}_pr_auc"] = overall["pr_auc"]
            row[f"{prefix}_recall"] = overall["recall"]
            row[f"{prefix}_f1"] = overall["f1"]
        row["delta_accuracy"] = row["age_specialist_accuracy"] - row["baseline_accuracy"]
        row["delta_roc_auc"] = row["age_specialist_roc_auc"] - row["baseline_roc_auc"]
        row["delta_pr_auc"] = row["age_specialist_pr_auc"] - row["baseline_pr_auc"]
        row["delta_recall"] = row["age_specialist_recall"] - row["baseline_recall"]
        row["delta_f1"] = row["age_specialist_f1"] - row["baseline_f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def print_model_summary(model_result: Dict[str, Any]) -> None:
    model_name = model_result["model_name"]
    baseline_overall = model_result["baseline"]["overall"]
    specialist_overall = model_result["age_specialist"]["overall"]
    baseline_group_metrics = model_result["baseline"]["group_metrics"]
    specialist_group_metrics = model_result["age_specialist"]["group_metrics"]
    baseline_model_info = model_result["baseline"]["model_info"]
    specialist_model_summary = model_result["age_specialist"]["model_info_by_group"]
    baseline_feature_info = model_result["baseline"]["feature_selection"]
    specialist_feature_summary = model_result["age_specialist"]["feature_selection_by_group"]
    baseline_sampling = model_result["baseline"]["sampling"]
    specialist_sampling = model_result["age_specialist"]["sampling_by_group"]

    print("\n" + "=" * 88)
    print(f"MODEL: {model_name}")
    print("=" * 88)
    print(f"{'Metric':<12} | {'Baseline':<14} | {'Age-Specialist':<14} | {'Winner':<14}")
    print("-" * 88)
    for metric in ["accuracy", "roc_auc", "pr_auc", "recall", "f1"]:
        baseline_value = baseline_overall[metric]
        specialist_value = specialist_overall[metric]
        winner = winner_for_metric(baseline_value, specialist_value)
        print(f"{metric:<12} | {fmt(baseline_value):<14} | {fmt(specialist_value):<14} | {winner:<14}")

    print("-" * 88)
    print(
        f"Baseline CM [TN FP FN TP]: "
        f"[{baseline_overall['tn']} {baseline_overall['fp']} {baseline_overall['fn']} {baseline_overall['tp']}]"
    )
    print(
        f"Age-Specialist CM [TN FP FN TP]: "
        f"[{specialist_overall['tn']} {specialist_overall['fp']} {specialist_overall['fn']} {specialist_overall['tp']}]"
    )

    print("\nPER-GROUP TEST ACCURACY")
    print(f"{'Age Group':<10} | {'Baseline':<14} | {'Age-Specialist':<14}")
    print("-" * 88)
    for group in AGE_GROUPS:
        print(
            f"{group:<10} | "
            f"{fmt(baseline_group_metrics[group]['accuracy']):<14} | "
            f"{fmt(specialist_group_metrics[group]['accuracy']):<14}"
        )

    print("\nHYPERPARAMETER TUNING")
    print("-" * 88)
    print(
        f"Baseline params: {baseline_model_info['params']} | "
        f"status: {baseline_model_info['status']} | "
        f"cv_best: {fmt(baseline_model_info['best_score'])}"
    )
    for group in AGE_GROUPS:
        model_info = specialist_model_summary[group]
        print(
            f"{group:<10} params: {model_info.get('params')} | "
            f"status: {model_info.get('status')} | "
            f"cv_best: {fmt(model_info.get('best_score'))}"
        )

    print("\nFEATURE SELECTION + SMOTE SUMMARY")
    print("-" * 88)
    print(f"Baseline top features: {format_feature_list(baseline_feature_info)}")
    print(
        f"Baseline SMOTE counts: before {baseline_sampling['before_counts']} "
        f"after {baseline_sampling['after_counts']}"
    )
    for group in AGE_GROUPS:
        print(f"{group:<10} features: {format_feature_list(specialist_feature_summary[group])}")
        sampling_info = specialist_sampling[group]
        if isinstance(sampling_info, dict):
            print(
                f"{group:<10} SMOTE: before {sampling_info['before_counts']} "
                f"after {sampling_info['after_counts']}"
            )
        else:
            print(f"{group:<10} SMOTE: N/A")


def run_model_comparison(
    model_key: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test_raw: pd.Series,
    feature_cols: list[str],
) -> Dict[str, Any]:
    model_name = get_model_display_name(model_key)

    baseline_result = fit_model_pipeline(
        train_df[feature_cols],
        train_df[TARGET_COL],
        test_df[feature_cols],
        label=f"{model_name} Baseline Holdout",
        model_key=model_key,
        top_k=FEATURE_SELECTION_TOP_K,
        use_smote=True,
        random_state=RANDOM_STATE,
        tune_hyperparameters=True,
    )
    baseline_overall = compute_metrics(y_test_raw, baseline_result["pred"], baseline_result["prob"])

    baseline_group_metrics = {}
    for group in AGE_GROUPS:
        group_mask = test_df["age_group"] == group
        baseline_group_metrics[group] = compute_metrics(
            test_df.loc[group_mask, TARGET_COL],
            baseline_result["pred"][group_mask.to_numpy()],
            baseline_result["prob"][group_mask.to_numpy()],
        )

    specialist_pred = pd.Series(index=test_df.index, dtype=float)
    specialist_prob = pd.Series(index=test_df.index, dtype=float)
    specialist_feature_summary: Dict[str, Dict[str, Any] | None] = {}
    specialist_sampling_summary: Dict[str, Dict[str, Any]] = {}
    specialist_model_summary: Dict[str, Dict[str, Any]] = {}

    baseline_pred_series = pd.Series(baseline_result["pred"], index=test_df.index)
    baseline_prob_series = pd.Series(baseline_result["prob"], index=test_df.index)

    for group in AGE_GROUPS:
        train_group_df = train_df[train_df["age_group"] == group]
        test_group_df = test_df[test_df["age_group"] == group]

        if test_group_df.empty:
            specialist_feature_summary[group] = None
            specialist_sampling_summary[group] = skipped_sampling_info(
                train_group_df[TARGET_COL].astype(int).reset_index(drop=True),
                "empty_test_cohort",
            )
            specialist_model_summary[group] = default_model_info(
                label=f"{model_name} {group} Holdout",
                reason="empty_test_cohort",
                status="skipped",
                model_key=model_key,
            )
            continue

        if train_group_df.empty or train_group_df[TARGET_COL].nunique() < 2:
            specialist_pred.loc[test_group_df.index] = baseline_pred_series.loc[test_group_df.index]
            specialist_prob.loc[test_group_df.index] = baseline_prob_series.loc[test_group_df.index]
            specialist_feature_summary[group] = None
            specialist_sampling_summary[group] = skipped_sampling_info(
                train_group_df[TARGET_COL].astype(int) if not train_group_df.empty else pd.Series(dtype=int),
                "insufficient_train_cohort",
            )
            specialist_model_summary[group] = default_model_info(
                label=f"{model_name} {group} Holdout",
                reason="insufficient_train_cohort",
                status="skipped",
                model_key=model_key,
            )
            continue

        group_result = fit_model_pipeline(
            train_group_df[feature_cols].reset_index(drop=True),
            train_group_df[TARGET_COL].astype(int).reset_index(drop=True),
            test_group_df[feature_cols].reset_index(drop=True),
            label=f"{model_name} {group} Holdout",
            model_key=model_key,
            top_k=FEATURE_SELECTION_TOP_K,
            use_smote=True,
            random_state=RANDOM_STATE,
            tune_hyperparameters=True,
        )
        specialist_pred.loc[test_group_df.index] = group_result["pred"]
        specialist_prob.loc[test_group_df.index] = group_result["prob"]
        specialist_feature_summary[group] = group_result["feature_selection"]
        specialist_sampling_summary[group] = group_result["sampling"]
        specialist_model_summary[group] = group_result["model_info"]

    missing_idx = specialist_pred[specialist_pred.isna()].index
    if len(missing_idx) > 0:
        specialist_pred.loc[missing_idx] = baseline_pred_series.loc[missing_idx]
        specialist_prob.loc[missing_idx] = baseline_prob_series.loc[missing_idx]

    specialist_pred = specialist_pred.astype(int)
    specialist_overall = compute_metrics(y_test_raw, specialist_pred.to_numpy(), specialist_prob.to_numpy())

    specialist_group_metrics = {}
    for group in AGE_GROUPS:
        group_indices = test_df.index[test_df["age_group"] == group]
        specialist_group_metrics[group] = compute_metrics(
            test_df.loc[group_indices, TARGET_COL],
            specialist_pred.loc[group_indices].to_numpy(),
            specialist_prob.loc[group_indices].to_numpy(),
        )

    return {
        "model_key": model_key,
        "model_name": model_name,
        "baseline": {
            "overall": baseline_overall,
            "group_metrics": baseline_group_metrics,
            "sampling": baseline_result["sampling"],
            "feature_selection": baseline_result["feature_selection"],
            "model_info": baseline_result["model_info"],
        },
        "age_specialist": {
            "overall": specialist_overall,
            "group_metrics": specialist_group_metrics,
            "sampling_by_group": specialist_sampling_summary,
            "feature_selection_by_group": specialist_feature_summary,
            "model_info_by_group": specialist_model_summary,
        },
    }


def compare_strategies():
    print("=" * 88)
    print("FIXED TRAIN/TEST HOLDOUT: BASELINE vs AGE-SPECIALIST")
    print(f"TOP-{FEATURE_SELECTION_TOP_K} FEATURE SELECTION + TRAIN-ONLY SMOTE")
    print("EVALUATION MODE: SINGLE FIXED TRAIN/TEST SPLIT")
    print("CLASSIFIERS: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, MLP, KNN,")
    print("             Decision Tree, Naive Bayes, CatBoost, AdaBoost")
    print("TUNING: RandomizedSearchCV on the training partition for every supported classifier")
    print("=" * 88)

    if not os.path.exists(RAW_FILE):
        print(f"Error: '{RAW_FILE}' not found.")
        return

    split_data = prepare_train_test_data(RAW_FILE, random_state=RANDOM_STATE)
    X_train_raw = split_data["X_train_raw"]
    X_test_raw = split_data["X_test_raw"]
    y_train_raw = split_data["y_train"]
    y_test_raw = split_data["y_test"]
    train_df = split_data["train_df"].copy()
    test_df = split_data["test_df"].copy()

    train_df["age_group"] = assign_age_group_raw(X_train_raw["age"])
    test_df["age_group"] = assign_age_group_raw(X_test_raw["age"])

    print(f"Raw cleaned rows: {len(train_df) + len(test_df)}")
    print(f"Train rows: {len(train_df)} | class counts: {y_train_raw.value_counts().sort_index().to_dict()}")
    print(f"Test rows: {len(test_df)} | class counts: {y_test_raw.value_counts().sort_index().to_dict()}")
    print(f"Train age groups: {train_df['age_group'].value_counts().to_dict()}")
    print(f"Test age groups: {test_df['age_group'].value_counts().to_dict()}")
    print("-" * 88)

    feature_cols = [column for column in train_df.columns if column not in [TARGET_COL, "age_group"]]
    available_models = get_available_model_keys()
    unavailable_models = get_unavailable_models()
    print(f"Available classifiers ({len(available_models)}): {', '.join(get_model_display_name(key) for key in available_models)}")
    if unavailable_models:
        print("Unavailable classifiers:")
        for model_key, reason in unavailable_models.items():
            print(f"- {get_model_display_name(model_key)}: {reason}")

    PERFORMANCE_VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_results: Dict[str, Dict[str, Any]] = {}
    for model_key in available_models:
        model_result = run_model_comparison(
            model_key=model_key,
            train_df=train_df,
            test_df=test_df,
            y_test_raw=y_test_raw,
            feature_cols=feature_cols,
        )
        model_result["visualizations"] = [
            str(path)
            for path in save_model_visualizations(
                model_key=model_key,
                model_label=model_result["model_name"],
                baseline_overall=model_result["baseline"]["overall"],
                specialist_overall=model_result["age_specialist"]["overall"],
                baseline_group_metrics=model_result["baseline"]["group_metrics"],
                specialist_group_metrics=model_result["age_specialist"]["group_metrics"],
                baseline_sampling=model_result["baseline"]["sampling"],
                specialist_sampling=model_result["age_specialist"]["sampling_by_group"],
                baseline_feature_info=model_result["baseline"]["feature_selection"],
                specialist_feature_selection=model_result["age_specialist"]["feature_selection_by_group"],
                baseline_model_info=model_result["baseline"]["model_info"],
                specialist_model_summary=model_result["age_specialist"]["model_info_by_group"],
            )
        ]
        comparison_results[model_key] = model_result
        print_model_summary(model_result)

        print("\nVISUALIZATION OUTPUTS")
        print("-" * 88)
        for visualization_path in model_result["visualizations"]:
            print(f"- {visualization_path}")

    cross_model_visuals = [str(path) for path in save_cross_model_visualizations(comparison_results)]
    summary_df = build_cross_model_summary(comparison_results)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

    results_payload = {
        "dataset": {
            "raw_file": str(RAW_FILE),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_class_counts": to_serializable(y_train_raw.value_counts().sort_index().to_dict()),
            "test_class_counts": to_serializable(y_test_raw.value_counts().sort_index().to_dict()),
            "train_age_groups": to_serializable(train_df["age_group"].value_counts().to_dict()),
            "test_age_groups": to_serializable(test_df["age_group"].value_counts().to_dict()),
        },
        "configuration": {
            "feature_selection_top_k": FEATURE_SELECTION_TOP_K,
            "smote_mode": "train_only",
            "holdout_random_state": RANDOM_STATE,
            "tuning_enabled": True,
            "available_models": available_models,
            "unavailable_models": unavailable_models,
        },
        "models": comparison_results,
        "cross_model_visualizations": cross_model_visuals,
        "cross_model_summary_csv": str(SUMMARY_CSV_PATH),
    }
    with RESULTS_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(results_payload), handle, indent=2)

    print("\n" + "=" * 88)
    print("CROSS-MODEL OUTPUTS")
    print("=" * 88)
    print(f"Results JSON: {RESULTS_JSON_PATH}")
    print(f"Summary CSV: {SUMMARY_CSV_PATH}")
    for visualization_path in cross_model_visuals:
        print(f"- {visualization_path}")


if __name__ == "__main__":
    ensure_root_artifact_dirs()
    compare_strategies()
