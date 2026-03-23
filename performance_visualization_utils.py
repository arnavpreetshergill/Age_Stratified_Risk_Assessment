from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from project_paths import PERFORMANCE_VISUALIZATIONS_DIR

AGE_GROUPS = ["Young", "Middle", "Elderly"]
OVERALL_METRICS: List[Tuple[str, str]] = [
    ("accuracy", "Accuracy"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("recall", "Recall"),
    ("f1", "F1"),
]
COHORT_METRICS: List[Tuple[str, str]] = [
    ("accuracy", "Accuracy"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
]
PALETTE = {
    "Baseline": "#355C7D",
    "Stratified": "#C06C4E",
}


def _safe_float(value: Any) -> float:
    if value is None:
        return np.nan
    return float(value)


def _count_value(counts: Dict[Any, Any], label: int) -> int:
    return int(counts.get(label, counts.get(str(label), 0)))


def _style() -> None:
    sns.set_theme(style="whitegrid")


def _sum_class_counts(sampling_map: Dict[str, Dict[str, Any]], key: str) -> Dict[int, int]:
    totals = {0: 0, 1: 0}
    for cohort in AGE_GROUPS:
        sampling_info = sampling_map.get(cohort)
        if not isinstance(sampling_info, dict):
            continue
        counts = sampling_info.get(key, {})
        totals[0] += _count_value(counts, 0)
        totals[1] += _count_value(counts, 1)
    return totals


def plot_overall_metrics(
    output_path: Path,
    baseline_overall: Dict[str, Any],
    strat_overall: Dict[str, Any],
) -> Path:
    _style()
    metric_labels = [label for _, label in OVERALL_METRICS]
    baseline_values = [_safe_float(baseline_overall.get(metric_key)) for metric_key, _ in OVERALL_METRICS]
    strat_values = [_safe_float(strat_overall.get(metric_key)) for metric_key, _ in OVERALL_METRICS]

    x = np.arange(len(metric_labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.bar(x - width / 2, baseline_values, width, label="Baseline", color=PALETTE["Baseline"])
    ax.bar(x + width / 2, strat_values, width, label="Stratified", color=PALETTE["Stratified"])
    ax.set_title("Overall Metric Comparison")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_per_cohort_metrics(
    output_path: Path,
    baseline_group_metrics: Dict[str, Dict[str, Any]],
    strat_group_metrics: Dict[str, Dict[str, Any]],
) -> Path:
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    x = np.arange(len(AGE_GROUPS))
    width = 0.34

    for index, (metric_key, metric_label) in enumerate(COHORT_METRICS):
        ax = axes[index]
        baseline_values = [
            _safe_float(baseline_group_metrics.get(cohort, {}).get(metric_key))
            for cohort in AGE_GROUPS
        ]
        strat_values = [
            _safe_float(strat_group_metrics.get(cohort, {}).get(metric_key))
            for cohort in AGE_GROUPS
        ]
        ax.bar(
            x - width / 2,
            baseline_values,
            width,
            label="Baseline",
            color=PALETTE["Baseline"],
        )
        ax.bar(
            x + width / 2,
            strat_values,
            width,
            label="Stratified",
            color=PALETTE["Stratified"],
        )
        ax.set_title(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(AGE_GROUPS)
        ax.set_ylim(0, 1.05)

    axes[len(COHORT_METRICS)].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Per-Cohort Metric Comparison", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrices(
    output_path: Path,
    baseline_overall: Dict[str, Any],
    strat_overall: Dict[str, Any],
) -> Path:
    _style()
    baseline_cm = np.array(
        [
            [baseline_overall["tn"], baseline_overall["fp"]],
            [baseline_overall["fn"], baseline_overall["tp"]],
        ]
    )
    strat_cm = np.array(
        [
            [strat_overall["tn"], strat_overall["fp"]],
            [strat_overall["fn"], strat_overall["tp"]],
        ]
    )
    vmax = max(baseline_cm.max(), strat_cm.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    sns.heatmap(
        baseline_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        vmin=0,
        vmax=vmax,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=axes[0],
    )
    axes[0].set_title("Baseline")
    sns.heatmap(
        strat_cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        cbar=False,
        vmin=0,
        vmax=vmax,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=axes[1],
    )
    axes[1].set_title("Stratified")
    fig.suptitle("Overall Confusion Matrix Comparison", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sampling_summary(
    output_path: Path,
    baseline_sampling: Dict[str, Any],
    strat_sampling: Dict[str, Dict[str, Any]],
) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    strat_before_counts = _sum_class_counts(strat_sampling, "before_counts")
    strat_after_counts = _sum_class_counts(strat_sampling, "after_counts")

    count_categories = ["Baseline Before", "Baseline After", "Stratified Before", "Stratified After"]
    class_0 = [
        _count_value(baseline_sampling["before_counts"], 0),
        _count_value(baseline_sampling["after_counts"], 0),
        strat_before_counts[0],
        strat_after_counts[0],
    ]
    class_1 = [
        _count_value(baseline_sampling["before_counts"], 1),
        _count_value(baseline_sampling["after_counts"], 1),
        strat_before_counts[1],
        strat_after_counts[1],
    ]

    x_counts = np.arange(len(count_categories))
    width = 0.34
    axes[0].bar(x_counts - width / 2, class_0, width, label="Class 0", color="#4C6A85")
    axes[0].bar(x_counts + width / 2, class_1, width, label="Class 1", color="#B86443")
    axes[0].set_title("Train Class Counts Before/After SMOTE")
    axes[0].set_xticks(x_counts)
    axes[0].set_xticklabels(count_categories, rotation=20, ha="right")
    axes[0].set_ylabel("Rows")
    axes[0].legend()

    sample_categories = ["Baseline Overall"] + AGE_GROUPS + ["Stratified Overall"]
    baseline_generated = int(
        sum(_count_value(baseline_sampling["after_counts"], label) for label in [0, 1])
        - sum(_count_value(baseline_sampling["before_counts"], label) for label in [0, 1])
    )
    strat_generated_per_cohort = [
        int(strat_sampling[cohort]["generated_samples"]) if strat_sampling.get(cohort) else 0
        for cohort in AGE_GROUPS
    ]
    generated_values = [baseline_generated] + strat_generated_per_cohort + [sum(strat_generated_per_cohort)]
    colors = [PALETTE["Baseline"]] + [PALETTE["Stratified"]] * (len(AGE_GROUPS) + 1)
    axes[1].bar(sample_categories, generated_values, color=colors)
    axes[1].set_title("Generated Samples Added by SMOTE")
    axes[1].set_ylabel("Generated rows")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("SMOTE Sampling Summary", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_selection(
    output_path: Path,
    baseline_feature_info: Dict[str, Any],
    strat_feature_selection: Dict[str, Dict[str, Any] | None],
) -> Path:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        ("Baseline", baseline_feature_info, PALETTE["Baseline"]),
        ("Young", strat_feature_selection.get("Young"), PALETTE["Stratified"]),
        ("Middle", strat_feature_selection.get("Middle"), PALETTE["Stratified"]),
        ("Elderly", strat_feature_selection.get("Elderly"), PALETTE["Stratified"]),
    ]

    for ax, (title, feature_info, color) in zip(axes.flatten(), panels):
        importances = feature_info.get("selected_feature_importances") if isinstance(feature_info, dict) else None
        if not isinstance(importances, dict) or len(importances) == 0:
            ax.text(0.5, 0.5, "No feature data", ha="center", va="center")
            ax.axis("off")
            continue

        feature_names = list(importances.keys())
        values = [float(importances[name]) for name in feature_names]
        order = np.argsort(values)
        sorted_features = [feature_names[index] for index in order]
        sorted_values = [values[index] for index in order]
        ax.barh(sorted_features, sorted_values, color=color)
        ax.set_title(title)
        ax.set_xlabel("Importance")

    fig.suptitle("Selected Feature Importance", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_visualizations(
    baseline_overall: Dict[str, Any],
    strat_overall: Dict[str, Any],
    baseline_group_metrics: Dict[str, Dict[str, Any]],
    strat_group_metrics: Dict[str, Dict[str, Any]],
    baseline_sampling: Dict[str, Any],
    strat_sampling: Dict[str, Dict[str, Any]],
    baseline_feature_info: Dict[str, Any],
    strat_feature_selection: Dict[str, Dict[str, Any] | None],
) -> List[Path]:
    PERFORMANCE_VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    outputs = [
        plot_overall_metrics(
            PERFORMANCE_VISUALIZATIONS_DIR / "overall_metrics_comparison.png",
            baseline_overall,
            strat_overall,
        ),
        plot_per_cohort_metrics(
            PERFORMANCE_VISUALIZATIONS_DIR / "per_cohort_metrics_comparison.png",
            baseline_group_metrics,
            strat_group_metrics,
        ),
        plot_confusion_matrices(
            PERFORMANCE_VISUALIZATIONS_DIR / "confusion_matrix_comparison.png",
            baseline_overall,
            strat_overall,
        ),
        plot_sampling_summary(
            PERFORMANCE_VISUALIZATIONS_DIR / "sampling_comparison.png",
            baseline_sampling,
            strat_sampling,
        ),
        plot_feature_selection(
            PERFORMANCE_VISUALIZATIONS_DIR / "feature_selection_comparison.png",
            baseline_feature_info,
            strat_feature_selection,
        ),
    ]
    return outputs
