from pathlib import Path
from typing import Any, Dict, List, Tuple

from shared_utils import AGE_GROUPS, run_variant

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


def main() -> None:
    root = Path(__file__).resolve().parent
    data_file = root / "heart_statlog_cleveland_hungary_final(1).csv"

    without_smote = run_variant(
        raw_file=data_file,
        use_smote=False,
        output_json=root / "without_smote" / "results_without_smote.json",
        variant_name="WITHOUT_SMOTE",
    )
    with_smote = run_variant(
        raw_file=data_file,
        use_smote=True,
        output_json=root / "with_smote" / "results_with_smote.json",
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
    print("MODEL PERFORMANCE COMPARISON (WITH SMOTE vs WITHOUT SMOTE) - COHORT-TUNED GRIDSEARCH")
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
    print("BEST HYPERPARAMETERS BY COHORT (GRIDSEARCH)")
    print("=" * 96)
    print(f"{'Cohort':<8} | {'Without SMOTE Best Params':<40} | {'With SMOTE Best Params'}")
    print("-" * 96)
    for cohort in AGE_GROUPS:
        without_tuning = without_smote["cohorts"][cohort]["tuning"]
        with_tuning = with_smote["cohorts"][cohort]["tuning"]
        without_params = params_to_text(without_tuning.get("best_params"))
        with_params = params_to_text(with_tuning.get("best_params"))
        print(f"{cohort:<8} | {without_params:<40} | {with_params}")

    report_lines = [
        "# SMOTE vs Non-SMOTE Cardiovascular Disease Comparison (Cohort-Tuned GridSearch)",
        "",
        "Target label meaning: `1 = cardiovascular disease`, `0 = no cardiovascular disease`.",
        "",
        "## Cohort rules",
        "",
    ]

    for cohort, rule in without_smote["cohort_definition"].items():
        report_lines.append(f"- `{cohort}`: {rule}")

    report_lines += [
        "",
        "## GridSearch configuration",
        "",
        f"- Method: {without_smote['grid_search']['method']}",
        f"- Scoring: {without_smote['grid_search']['scoring']}",
        f"- Param grid: {without_smote['grid_search']['param_grid']}",
        f"- Grid size per cohort: {without_smote['grid_search']['param_grid_size']}",
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
        "## Best hyperparameters by cohort",
        "",
        "| Cohort | Without SMOTE | With SMOTE |",
        "|---|---|---|",
    ]

    for cohort in AGE_GROUPS:
        without_tuning = without_smote["cohorts"][cohort]["tuning"]
        with_tuning = with_smote["cohorts"][cohort]["tuning"]
        without_params = params_to_text(without_tuning.get("best_params"))
        with_params = params_to_text(with_tuning.get("best_params"))
        report_lines.append(f"| {cohort} | {without_params} | {with_params} |")

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
    ]

    report_path = root / "SMOTE_COMPARISON_REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
