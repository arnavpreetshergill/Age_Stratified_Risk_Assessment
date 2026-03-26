from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline_utils import (
    AGE_COL,
    AGE_GROUPS,
    COHORT_RULES,
    TARGET_COL,
    assert_frames_match,
    assign_age_group_processed,
    get_processed_age_cutoffs,
    prepare_train_test_data,
)
from holdout_utils import default_model_info, fit_xgboost_pipeline
from training_utils import FEATURE_SELECTION_TOP_K, default_model_params

RANDOM_STATE = 42


def class_counts(y: pd.Series) -> Dict[int, int]:
    counts = y.astype(int).value_counts().sort_index()
    count_map = {0: 0, 1: 0}
    for label, count in counts.items():
        count_map[int(label)] = int(count)
    return count_map


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "pr_auc": None,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }

    if np.unique(y_true_arr).size > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
        metrics["pr_auc"] = float(average_precision_score(y_true_arr, y_prob_arr))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    return metrics


def skipped_sampling_info(y_train: pd.Series, reason: str) -> Dict[str, Any]:
    count_map = class_counts(y_train)
    return {
        "applied": False,
        "reason": reason,
        "before_counts": count_map,
        "after_counts": count_map.copy(),
        "generated_samples": 0,
        "k_neighbors_used": 0,
    }


def _to_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        float_value = float(value)
        return None if np.isnan(float_value) else float_value
    return value


def save_results(output_json: Path | None, results: Dict[str, Any]) -> None:
    if output_json is None:
        return

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(_to_json_ready(results), indent=2),
        encoding="utf-8",
    )


def print_variant_summary(results: Dict[str, Any]) -> None:
    def _fmt_metric(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.4f}"

    variant_name = results["variant"]
    overall_before_counts = results["train_class_counts_before_sampling_overall"]
    overall_after_counts = results["train_class_counts_after_sampling_overall"]

    print(f"[{variant_name}] Train counts before sampling: {overall_before_counts}")
    print(f"[{variant_name}] Train counts after sampling:  {overall_after_counts}")
    print(f"[{variant_name}] Test rows: {results['test_rows']}")
    for cohort in AGE_GROUPS:
        cohort_data = results["cohorts"].get(cohort, {})
        cohort_metrics = cohort_data.get("metrics")
        if cohort_metrics is None:
            print(f"[{variant_name}] {cohort}: metrics unavailable")
            continue
        model_params = cohort_data.get("model_info", {}).get("params")
        print(
            f"[{variant_name}] {cohort}: "
            f"Acc={_fmt_metric(cohort_metrics['accuracy'])}, "
            f"Recall={_fmt_metric(cohort_metrics['recall'])}, "
            f"F1={_fmt_metric(cohort_metrics['f1'])}, "
            f"ROC-AUC={_fmt_metric(cohort_metrics['roc_auc'])}, "
            f"Params={model_params}"
        )

    overall_metrics = results["overall_metrics"]
    print(
        f"[{variant_name}] OVERALL: "
        f"Accuracy={_fmt_metric(overall_metrics['accuracy'])}, "
        f"Recall={_fmt_metric(overall_metrics['recall'])}, "
        f"F1={_fmt_metric(overall_metrics['f1'])}, "
        f"ROC-AUC={_fmt_metric(overall_metrics['roc_auc'])}"
    )


def _validate_holdout_artifacts(
    split_data: Dict[str, Any],
    processed_train_df: pd.DataFrame,
    processed_test_df: pd.DataFrame,
) -> None:
    assert_frames_match(split_data["test_df"], processed_test_df, "processed_test.csv")
    if list(processed_train_df.columns) != list(split_data["train_df"].columns):
        raise ValueError(
            "processed_train_100k_stratified.csv columns do not match the canonical processed-train schema."
        )


def run_variant_on_generated_data(
    raw_file: Path,
    generated_train_file: Path,
    processed_test_file: Path,
    use_smote: bool,
    output_json: Path | None,
    variant_name: str,
) -> Dict[str, Any]:
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_file}")
    if not generated_train_file.exists():
        raise FileNotFoundError(f"Generated train file not found: {generated_train_file}")
    if not processed_test_file.exists():
        raise FileNotFoundError(f"Processed test file not found: {processed_test_file}")

    split_data = prepare_train_test_data(raw_file, random_state=RANDOM_STATE)
    age_z_45, age_z_65 = get_processed_age_cutoffs(split_data["preprocessor"])

    generated_train_df = pd.read_csv(generated_train_file)
    processed_test_df = pd.read_csv(processed_test_file)
    _validate_holdout_artifacts(split_data, generated_train_df, processed_test_df)

    train_age_groups = assign_age_group_processed(
        generated_train_df[AGE_COL],
        young_upper_bound=age_z_45,
        middle_upper_bound=age_z_65,
    )
    test_age_groups = assign_age_group_processed(
        processed_test_df[AGE_COL],
        young_upper_bound=age_z_45,
        middle_upper_bound=age_z_65,
    )
    feature_cols = [column for column in generated_train_df.columns if column != TARGET_COL]

    final_pred = pd.Series(index=processed_test_df.index, dtype=float)
    final_prob = pd.Series(index=processed_test_df.index, dtype=float)
    feature_summary: Dict[str, Any] = {}
    sampling_summary: Dict[str, Any] = {}
    model_summary: Dict[str, Any] = {}
    train_rows_before_by_cohort: Dict[str, int] = {}
    train_rows_after_by_cohort: Dict[str, int] = {}
    missing_prediction_fallback = 0

    total_before_counts = class_counts(generated_train_df[TARGET_COL].astype(int))
    total_after_counts = {0: 0, 1: 0}

    print(
        f"[{variant_name}] Using generated training data {generated_train_file.name} "
        f"({len(generated_train_df)} rows) and test data {processed_test_file.name} "
        f"({len(processed_test_df)} rows)"
    )
    print(f"[{variant_name}] Test age groups: {test_age_groups.value_counts().to_dict()}")

    for cohort in AGE_GROUPS:
        train_mask = train_age_groups == cohort
        test_mask = test_age_groups == cohort

        X_train_cohort = generated_train_df.loc[train_mask, feature_cols].reset_index(drop=True)
        y_train_cohort = generated_train_df.loc[train_mask, TARGET_COL].astype(int).reset_index(drop=True)
        X_test_cohort = processed_test_df.loc[test_mask, feature_cols].reset_index(drop=True)

        train_rows_before_by_cohort[cohort] = int(len(y_train_cohort))

        if len(X_test_cohort) == 0:
            feature_summary[cohort] = None
            sampling_summary[cohort] = skipped_sampling_info(y_train_cohort, "empty_test_cohort")
            model_summary[cohort] = default_model_info(
                label=f"{variant_name} {cohort} Holdout",
                reason="empty_test_cohort",
                status="skipped",
            )
            train_rows_after_by_cohort[cohort] = int(len(y_train_cohort))
            continue

        if len(X_train_cohort) == 0 or y_train_cohort.nunique() < 2:
            fallback_prob = float(generated_train_df[TARGET_COL].astype(int).mean())
            fallback_pred = int(fallback_prob >= 0.5)
            cohort_pred = np.full(len(X_test_cohort), fallback_pred, dtype=int)
            cohort_prob = np.full(len(X_test_cohort), fallback_prob, dtype=float)
            feature_summary[cohort] = None
            sampling_summary[cohort] = skipped_sampling_info(y_train_cohort, "insufficient_train_cohort")
            model_summary[cohort] = default_model_info(
                label=f"{variant_name} {cohort} Holdout",
                reason="insufficient_train_cohort",
                status="skipped",
            )
            train_rows_after_by_cohort[cohort] = int(len(y_train_cohort))
        else:
            cohort_result = fit_xgboost_pipeline(
                X_train_cohort,
                y_train_cohort,
                X_test_cohort,
                label=f"{variant_name} {cohort} Holdout",
                top_k=FEATURE_SELECTION_TOP_K,
                use_smote=use_smote,
                random_state=RANDOM_STATE,
            )
            cohort_pred = cohort_result["pred"]
            cohort_prob = cohort_result["prob"]
            feature_summary[cohort] = cohort_result["feature_selection"]
            sampling_summary[cohort] = cohort_result["sampling"]
            model_summary[cohort] = cohort_result["model_info"]
            after_counts = cohort_result["sampling"].get("after_counts", {})
            train_rows_after_by_cohort[cohort] = int(
                after_counts.get(0, after_counts.get("0", 0))
                + after_counts.get(1, after_counts.get("1", 0))
            )

        cohort_indices = processed_test_df.index[test_mask]
        final_pred.loc[cohort_indices] = cohort_pred
        final_prob.loc[cohort_indices] = cohort_prob

        sampling_info = sampling_summary[cohort]
        if isinstance(sampling_info, dict):
            after_counts = sampling_info.get("after_counts", {})
            total_after_counts[0] += int(after_counts.get(0, after_counts.get("0", 0)))
            total_after_counts[1] += int(after_counts.get(1, after_counts.get("1", 0)))

    missing_idx = final_pred[final_pred.isna()].index
    if len(missing_idx) > 0:
        fallback_prob = float(generated_train_df[TARGET_COL].astype(int).mean())
        fallback_pred = int(fallback_prob >= 0.5)
        final_pred.loc[missing_idx] = fallback_pred
        final_prob.loc[missing_idx] = fallback_prob
        missing_prediction_fallback = int(len(missing_idx))

    final_pred = final_pred.astype(int)
    overall_metrics = compute_metrics(
        processed_test_df[TARGET_COL].astype(int),
        final_pred.to_numpy(),
        final_prob.to_numpy(),
    )

    cohort_results: Dict[str, Any] = {}
    for cohort in AGE_GROUPS:
        mask = test_age_groups == cohort
        cohort_results[cohort] = {
            "rule": COHORT_RULES[cohort],
            "train_rows_before_sampling": train_rows_before_by_cohort[cohort],
            "train_rows_after_sampling": train_rows_after_by_cohort[cohort],
            "test_rows": int(mask.sum()),
            "feature_selection": feature_summary[cohort],
            "sampling": sampling_summary[cohort],
            "model_info": model_summary[cohort],
            "metrics": compute_metrics(
                processed_test_df.loc[mask, TARGET_COL].astype(int),
                final_pred.loc[mask].to_numpy(),
                final_prob.loc[mask].to_numpy(),
            ),
        }

    results: Dict[str, Any] = {
        "variant": variant_name,
        "use_smote": bool(use_smote),
        "dataset_rows_after_cleaning": int(len(split_data["train_df"]) + len(split_data["test_df"])),
        "dataset_source": {
            "mode": "fixed_generated_train_with_real_holdout_test",
            "raw_file": str(raw_file),
            "generated_train_file": str(generated_train_file),
            "processed_test_file": str(processed_test_file),
            "generated_train_rows": int(len(generated_train_df)),
            "test_source": "fixed processed_test.csv transformed with the train-fitted preprocessor",
            "processed_age_cutoffs": "derived once from the single train-fitted scaler used for processed_train/processed_test",
        },
        "feature_selection": {
            "enabled": True,
            "method": "xgboost_feature_importance_top_k",
            "top_k": int(FEATURE_SELECTION_TOP_K),
            "stage": "before the single holdout fit on each cohort",
        },
        "target_definition": "1 = cardiovascular disease, 0 = no cardiovascular disease",
        "cohort_definition": COHORT_RULES,
        "model_configuration": {
            "method": "default_xgboost_params",
            "selection_policy": "fixed_default_values",
            "params": default_model_params(),
        },
        "train_rows_before_sampling": int(len(generated_train_df)),
        "train_rows_after_sampling": int(total_after_counts[0] + total_after_counts[1]),
        "test_rows": int(len(processed_test_df)),
        "train_class_counts_before_sampling_overall": total_before_counts,
        "train_class_counts_after_sampling_overall": total_after_counts,
        "cohorts": cohort_results,
        "overall_metrics": overall_metrics,
        "missing_prediction_fallback": {
            "rows_filled": int(missing_prediction_fallback),
            "fallback": "generated-train prevalence",
        } if missing_prediction_fallback > 0 else None,
    }

    save_results(output_json, results)
    print_variant_summary(results)
    return results


def run_variant(
    raw_file: Path,
    generated_train_file: Path,
    processed_test_file: Path,
    use_smote: bool,
    output_json: Path | None,
    variant_name: str,
) -> Dict[str, Any]:
    return run_variant_on_generated_data(
        raw_file=raw_file,
        generated_train_file=generated_train_file,
        processed_test_file=processed_test_file,
        use_smote=use_smote,
        output_json=output_json,
        variant_name=variant_name,
    )
