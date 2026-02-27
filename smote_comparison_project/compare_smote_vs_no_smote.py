from pathlib import Path
from typing import Any, Dict, List, Tuple

from shared_utils import run_variant

METRICS: List[Tuple[str, str]] = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
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

    rows = []
    for metric_key, metric_label in METRICS:
        base_value = without_smote["metrics"].get(metric_key)
        smote_value = with_smote["metrics"].get(metric_key)

        if base_value is None or smote_value is None:
            delta = None
            winner = "N/A"
        else:
            delta = float(smote_value) - float(base_value)
            if delta > 0:
                winner = "With SMOTE"
            elif delta < 0:
                winner = "Without SMOTE"
            else:
                winner = "Tie"

        rows.append((metric_label, base_value, smote_value, delta, winner))

    print("=" * 84)
    print("MODEL PERFORMANCE COMPARISON (WITH SMOTE vs WITHOUT SMOTE)")
    print("=" * 84)
    print(f"{'Metric':<12} | {'Without SMOTE':<13} | {'With SMOTE':<10} | {'Delta':<10} | Winner")
    print("-" * 84)
    for metric_label, base_value, smote_value, delta, winner in rows:
        print(
            f"{metric_label:<12} | {fmt(base_value):<13} | {fmt(smote_value):<10} | {fmt_delta(delta):<10} | {winner}"
        )

    base_cm = without_smote["metrics"]
    smote_cm = with_smote["metrics"]

    print("-" * 84)
    print(
        "Without SMOTE CM [TN FP FN TP]: "
        f"[{base_cm['tn']} {base_cm['fp']} {base_cm['fn']} {base_cm['tp']}]"
    )
    print(
        "With SMOTE CM [TN FP FN TP]:    "
        f"[{smote_cm['tn']} {smote_cm['fp']} {smote_cm['fn']} {smote_cm['tp']}]"
    )

    report_lines = [
        "# SMOTE vs Non-SMOTE Cardiovascular Disease Model Comparison",
        "",
        "Target label meaning: `1 = cardiovascular disease`, `0 = no cardiovascular disease`.",
        "",
        "## Train class distribution",
        "",
        f"- Without SMOTE (before): {without_smote['sampling']['before_counts']}",
        f"- Without SMOTE (after):  {without_smote['sampling']['after_counts']}",
        f"- With SMOTE (before):    {with_smote['sampling']['before_counts']}",
        f"- With SMOTE (after):     {with_smote['sampling']['after_counts']}",
        "",
        "## Performance summary",
        "",
        "| Metric | Without SMOTE | With SMOTE | Delta (With - Without) | Better |",
        "|---|---:|---:|---:|---|",
    ]

    for metric_label, base_value, smote_value, delta, winner in rows:
        report_lines.append(
            f"| {metric_label} | {fmt(base_value)} | {fmt(smote_value)} | {fmt_delta(delta)} | {winner} |"
        )

    report_lines += [
        "",
        "## Confusion matrices",
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
