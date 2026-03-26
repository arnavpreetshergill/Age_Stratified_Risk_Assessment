from typing import Any, Dict, List, Tuple

from project_paths import (
    REPORT_FILE,
    ROOT_GENERATED_TRAIN_FILE,
    ROOT_PROCESSED_TEST_FILE,
    RESULTS_WITHOUT_SMOTE_FILE,
    RESULTS_WITH_SMOTE_FILE,
    ROOT_RAW_DATA_FILE,
    ensure_subproject_artifact_dirs,
)
from shared_utils import AGE_GROUPS, run_variant_on_generated_data
from visualization_utils import save_visualizations

OVERALL_METRICS: List[Tuple[str, str]] = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
]

COHORT_METRICS: List[Tuple[str, str]] = [
    ("accuracy", "Accuracy"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
]


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def fmt_delta(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):+.4f}"


def pick_winner(base_value: Any, smote_value: Any) -> str:
    if base_value is None or smote_value is None:
        return "N/A"
    delta = float(smote_value) - float(base_value)
    if delta > 0:
        return "With SMOTE"
    if delta < 0:
        return "Without SMOTE"
    return "Tie"


def params_to_text(params: Any) -> str:
    if not isinstance(params, dict) or len(params) == 0:
        return "N/A"
    ordered_keys = ["n_estimators", "max_depth", "learning_rate", "subsample"]
    text_parts = []
    for key in ordered_keys:
        if key in params:
            text_parts.append(f"{key}={params[key]}")
    for key in params:
        if key not in ordered_keys:
            text_parts.append(f"{key}={params[key]}")
    return ", ".join(text_parts)


def features_to_text(feature_selection: Dict[str, Any] | None) -> str:
    if not isinstance(feature_selection, dict):
        return "N/A"
    selected_features = feature_selection.get("selected_features")
    if not isinstance(selected_features, list) or len(selected_features) == 0:
        return "N/A"
    return ", ".join(str(feature_name) for feature_name in selected_features)


def main() -> None:
    ensure_subproject_artifact_dirs()
    raw_file = ROOT_RAW_DATA_FILE

    without_smote = run_variant_on_generated_data(
        raw_file=raw_file,
        generated_train_file=ROOT_GENERATED_TRAIN_FILE,
        processed_test_file=ROOT_PROCESSED_TEST_FILE,
        use_smote=False,
        output_json=RESULTS_WITHOUT_SMOTE_FILE,
        variant_name="WITHOUT_SMOTE",
    )
    with_smote = run_variant_on_generated_data(
        raw_file=raw_file,
        generated_train_file=ROOT_GENERATED_TRAIN_FILE,
        processed_test_file=ROOT_PROCESSED_TEST_FILE,
        use_smote=True,
        output_json=RESULTS_WITH_SMOTE_FILE,
        variant_name="WITH_SMOTE",
    )

    overall_rows = []
    for metric_key, metric_label in OVERALL_METRICS:
        base_value = without_smote["overall_metrics"].get(metric_key)
        smote_value = with_smote["overall_metrics"].get(metric_key)
        delta = None if base_value is None or smote_value is None else float(smote_value) - float(base_value)
        winner = pick_winner(base_value, smote_value)
        overall_rows.append((metric_label, base_value, smote_value, delta, winner))

    print("=" * 96)
    print("MODEL PERFORMANCE COMPARISON (WITH SMOTE vs WITHOUT SMOTE) - FIXED TRAIN/TEST HOLDOUT")
    print("=" * 96)
    print(f"{'Metric':<12} | {'Without SMOTE':<13} | {'With SMOTE':<10} | {'Delta':<10} | Winner")
    print("-" * 96)
    for metric_label, base_value, smote_value, delta, winner in overall_rows:
        print(
            f"{metric_label:<12} | {fmt(base_value):<13} | {fmt(smote_value):<10} | {fmt_delta(delta):<10} | {winner}"
        )

    base_cm = without_smote["overall_metrics"]
    smote_cm = with_smote["overall_metrics"]
    print("-" * 96)
    print(
        "Without SMOTE OVERALL CM [TN FP FN TP]: "
        f"[{base_cm['tn']} {base_cm['fp']} {base_cm['fn']} {base_cm['tp']}]"
    )
    print(
        "With SMOTE OVERALL CM [TN FP FN TP]:    "
        f"[{smote_cm['tn']} {smote_cm['fp']} {smote_cm['fn']} {smote_cm['tp']}]"
    )

    print("\n" + "=" * 96)
    print("PER-COHORT METRICS")
    print("=" * 96)
    print(
        f"{'Cohort':<8} | {'Metric':<10} | {'Without SMOTE':<13} | {'With SMOTE':<10} | {'Delta':<10} | Winner"
    )
    print("-" * 96)

    cohort_rows = []
    for cohort in AGE_GROUPS:
        without_metrics = without_smote["cohorts"][cohort]["metrics"]
        with_metrics = with_smote["cohorts"][cohort]["metrics"]
        for metric_key, metric_label in COHORT_METRICS:
            base_value = without_metrics.get(metric_key) if without_metrics else None
            smote_value = with_metrics.get(metric_key) if with_metrics else None
            delta = None if base_value is None or smote_value is None else float(smote_value) - float(base_value)
            winner = pick_winner(base_value, smote_value)
            cohort_rows.append((cohort, metric_label, base_value, smote_value, delta, winner))
            print(
                f"{cohort:<8} | {metric_label:<10} | {fmt(base_value):<13} | {fmt(smote_value):<10} | {fmt_delta(delta):<10} | {winner}"
            )

    print("\n" + "=" * 96)
    print("MODEL PARAMETERS BY COHORT")
    print("=" * 96)
    print(f"{'Cohort':<8} | {'Without SMOTE Params':<40} | {'With SMOTE Params'}")
    print("-" * 96)
    for cohort in AGE_GROUPS:
        without_model_info = without_smote["cohorts"][cohort]["model_info"]
        with_model_info = with_smote["cohorts"][cohort]["model_info"]
        without_params = params_to_text(without_model_info.get("params"))
        with_params = params_to_text(with_model_info.get("params"))
        print(f"{cohort:<8} | {without_params:<40} | {with_params}")

    print("\n" + "=" * 96)
    print("SELECTED TOP FEATURES BY COHORT")
    print("=" * 96)
    print(f"{'Cohort':<8} | {'Without SMOTE Top Features':<60} | {'With SMOTE Top Features'}")
    print("-" * 96)
    for cohort in AGE_GROUPS:
        without_features = features_to_text(
            without_smote["cohorts"][cohort].get("feature_selection")
        )
        with_features = features_to_text(
            with_smote["cohorts"][cohort].get("feature_selection")
        )
        print(f"{cohort:<8} | {without_features:<60} | {with_features}")

    visualization_paths = save_visualizations(without_smote, with_smote)

    report_lines = [
        "# SMOTE vs Non-SMOTE Cardiovascular Disease Comparison (Fixed Train/Test Holdout)",
        "",
        "Target label meaning: `1 = cardiovascular disease`, `0 = no cardiovascular disease`.",
        "",
        "## Data sources",
        "",
        f"- Raw dataset: `{without_smote['dataset_source']['raw_file']}`",
        f"- Evaluation mode: `{without_smote['dataset_source']['mode']}`",
        f"- Generated train file: `{without_smote['dataset_source']['generated_train_file']}`",
        f"- Processed test file: `{without_smote['dataset_source']['processed_test_file']}`",
        f"- Generated train rows: `{without_smote['dataset_source']['generated_train_rows']}`",
        f"- Test source: {without_smote['dataset_source']['test_source']}",
        f"- Processed age cutoffs: {without_smote['dataset_source']['processed_age_cutoffs']}",
        "",
        "## Cohort rules",
        "",
    ]

    for cohort, rule in without_smote["cohort_definition"].items():
        report_lines.append(f"- `{cohort}`: {rule}")

    report_lines += [
        "",
        "## Model parameter configuration",
        "",
        f"- Method: {without_smote['model_configuration']['method']}",
        f"- Selection policy: {without_smote['model_configuration']['selection_policy']}",
        f"- Params: {without_smote['model_configuration']['params']}",
        "",
        "## Feature selection configuration",
        "",
        f"- Method: {without_smote['feature_selection']['method']}",
        f"- Top K: {without_smote['feature_selection']['top_k']}",
        f"- Stage: {without_smote['feature_selection']['stage']}",
        "",
        "## Overall train class distribution",
        "",
        f"- Without SMOTE (before): {without_smote['train_class_counts_before_sampling_overall']}",
        f"- Without SMOTE (after):  {without_smote['train_class_counts_after_sampling_overall']}",
        f"- With SMOTE (before):    {with_smote['train_class_counts_before_sampling_overall']}",
        f"- With SMOTE (after):     {with_smote['train_class_counts_after_sampling_overall']}",
        "",
        "## Overall performance summary",
        "",
        "| Metric | Without SMOTE | With SMOTE | Delta (With - Without) | Better |",
        "|---|---:|---:|---:|---|",
    ]

    for metric_label, base_value, smote_value, delta, winner in overall_rows:
        report_lines.append(
            f"| {metric_label} | {fmt(base_value)} | {fmt(smote_value)} | {fmt_delta(delta)} | {winner} |"
        )

    report_lines += [
        "",
        "## Per-cohort performance summary",
        "",
        "| Cohort | Metric | Without SMOTE | With SMOTE | Delta (With - Without) | Better |",
        "|---|---|---:|---:|---:|---|",
    ]

    for cohort, metric_label, base_value, smote_value, delta, winner in cohort_rows:
        report_lines.append(
            f"| {cohort} | {metric_label} | {fmt(base_value)} | {fmt(smote_value)} | {fmt_delta(delta)} | {winner} |"
        )

    report_lines += [
        "",
        "## Evaluation split details",
        "",
        f"- Generated training rows: `{without_smote['train_rows_before_sampling']}`",
        f"- Test rows: `{without_smote['test_rows']}`",
        f"- Without SMOTE train rows after sampling: `{without_smote['train_rows_after_sampling']}`",
        f"- With SMOTE train rows after sampling: `{with_smote['train_rows_after_sampling']}`",
        "",
        "## Model parameters by cohort",
        "",
        "| Cohort | Without SMOTE | With SMOTE |",
        "|---|---|---|",
    ]

    for cohort in AGE_GROUPS:
        without_model_info = without_smote["cohorts"][cohort]["model_info"]
        with_model_info = with_smote["cohorts"][cohort]["model_info"]
        without_params = params_to_text(without_model_info.get("params"))
        with_params = params_to_text(with_model_info.get("params"))
        report_lines.append(f"| {cohort} | {without_params} | {with_params} |")

    report_lines += [
        "",
        "## Selected top features by cohort",
        "",
        "| Cohort | Without SMOTE | With SMOTE |",
        "|---|---|---|",
    ]

    for cohort in AGE_GROUPS:
        without_features = features_to_text(
            without_smote["cohorts"][cohort].get("feature_selection")
        )
        with_features = features_to_text(
            with_smote["cohorts"][cohort].get("feature_selection")
        )
        report_lines.append(f"| {cohort} | {without_features} | {with_features} |")

    report_lines += [
        "",
        "## Overall confusion matrices",
        "",
        (
            "- Without SMOTE [TN FP FN TP]: "
            f"[{base_cm['tn']} {base_cm['fp']} {base_cm['fn']} {base_cm['tp']}]"
        ),
        (
            "- With SMOTE [TN FP FN TP]: "
            f"[{smote_cm['tn']} {smote_cm['fp']} {smote_cm['fn']} {smote_cm['tp']}]"
        ),
        "",
        "## Visualization outputs",
        "",
    ]

    for visualization_path in visualization_paths:
        report_lines.append(f"- `{visualization_path.name}`")

    report_lines.append("")

    report_path = REPORT_FILE
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nSaved report: {report_path}")
    print("Saved visualizations:")
    for visualization_path in visualization_paths:
        print(f"- {visualization_path}")


if __name__ == "__main__":
    main()
