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
    "Age-Specialist": "#C06C4E",
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


def _model_dir(model_key: str) -> Path:
    return PERFORMANCE_VISUALIZATIONS_DIR / model_key


def _summary_dir() -> Path:
    return PERFORMANCE_VISUALIZATIONS_DIR / "_summary"


def plot_overall_metrics(
    output_path: Path,
    baseline_overall: Dict[str, Any],
    specialist_overall: Dict[str, Any],
    model_label: str,
) -> Path:
    _style()
    metric_labels = [label for _, label in OVERALL_METRICS]
    baseline_values = [_safe_float(baseline_overall.get(metric_key)) for metric_key, _ in OVERALL_METRICS]
    specialist_values = [_safe_float(specialist_overall.get(metric_key)) for metric_key, _ in OVERALL_METRICS]

    x = np.arange(len(metric_labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.bar(x - width / 2, baseline_values, width, label="Baseline", color=PALETTE["Baseline"])
    ax.bar(x + width / 2, specialist_values, width, label="Age-Specialist", color=PALETTE["Age-Specialist"])
    ax.set_title(f"{model_label}: Holdout Overall Metric Comparison")
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
    specialist_group_metrics: Dict[str, Dict[str, Any]],
    model_label: str,
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
        specialist_values = [
            _safe_float(specialist_group_metrics.get(cohort, {}).get(metric_key))
            for cohort in AGE_GROUPS
        ]
        ax.bar(x - width / 2, baseline_values, width, label="Baseline", color=PALETTE["Baseline"])
        ax.bar(x + width / 2, specialist_values, width, label="Age-Specialist", color=PALETTE["Age-Specialist"])
        ax.set_title(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(AGE_GROUPS)
        ax.set_ylim(0, 1.05)

    axes[len(COHORT_METRICS)].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"{model_label}: Per-Cohort Holdout Metrics", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrices(
    output_path: Path,
    baseline_overall: Dict[str, Any],
    specialist_overall: Dict[str, Any],
    model_label: str,
) -> Path:
    _style()
    baseline_cm = np.array(
        [
            [baseline_overall["tn"], baseline_overall["fp"]],
            [baseline_overall["fn"], baseline_overall["tp"]],
        ]
    )
    specialist_cm = np.array(
        [
            [specialist_overall["tn"], specialist_overall["fp"]],
            [specialist_overall["fn"], specialist_overall["tp"]],
        ]
    )
    vmax = max(baseline_cm.max(), specialist_cm.max())

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
        specialist_cm,
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
    axes[1].set_title("Age-Specialist")
    fig.suptitle(f"{model_label}: Holdout Confusion Matrix Comparison", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sampling_summary(
    output_path: Path,
    baseline_sampling: Dict[str, Any],
    specialist_sampling: Dict[str, Dict[str, Any]],
    model_label: str,
) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    specialist_before_counts = _sum_class_counts(specialist_sampling, "before_counts")
    specialist_after_counts = _sum_class_counts(specialist_sampling, "after_counts")

    count_categories = ["Baseline Before", "Baseline After", "Age-Specialist Before", "Age-Specialist After"]
    class_0 = [
        _count_value(baseline_sampling["before_counts"], 0),
        _count_value(baseline_sampling["after_counts"], 0),
        specialist_before_counts[0],
        specialist_after_counts[0],
    ]
    class_1 = [
        _count_value(baseline_sampling["before_counts"], 1),
        _count_value(baseline_sampling["after_counts"], 1),
        specialist_before_counts[1],
        specialist_after_counts[1],
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

    sample_categories = ["Baseline Overall"] + AGE_GROUPS + ["Age-Specialist Overall"]
    baseline_generated = int(
        sum(_count_value(baseline_sampling["after_counts"], label) for label in [0, 1])
        - sum(_count_value(baseline_sampling["before_counts"], label) for label in [0, 1])
    )
    specialist_generated_per_cohort = [
        int(specialist_sampling[cohort]["generated_samples"]) if specialist_sampling.get(cohort) else 0
        for cohort in AGE_GROUPS
    ]
    generated_values = [baseline_generated] + specialist_generated_per_cohort + [sum(specialist_generated_per_cohort)]
    colors = [PALETTE["Baseline"]] + [PALETTE["Age-Specialist"]] * (len(AGE_GROUPS) + 1)
    axes[1].bar(sample_categories, generated_values, color=colors)
    axes[1].set_title("Generated Samples Added by SMOTE")
    axes[1].set_ylabel("Generated rows")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle(f"{model_label}: Train-Only SMOTE Summary", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_selection(
    output_path: Path,
    baseline_feature_info: Dict[str, Any],
    specialist_feature_selection: Dict[str, Dict[str, Any] | None],
    model_label: str,
) -> Path:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        ("Baseline", baseline_feature_info, PALETTE["Baseline"]),
        ("Young", specialist_feature_selection.get("Young"), PALETTE["Age-Specialist"]),
        ("Middle", specialist_feature_selection.get("Middle"), PALETTE["Age-Specialist"]),
        ("Elderly", specialist_feature_selection.get("Elderly"), PALETTE["Age-Specialist"]),
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

    fig.suptitle(f"{model_label}: Selected Feature Importance", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_tuning_summary(
    output_path: Path,
    baseline_model_info: Dict[str, Any],
    specialist_model_summary: Dict[str, Dict[str, Any]],
    model_label: str,
) -> Path:
    _style()
    entries = [("Baseline", baseline_model_info)] + [
        (cohort, specialist_model_summary.get(cohort, {}))
        for cohort in AGE_GROUPS
    ]
    scores = [_safe_float(info.get("best_score")) for _, info in entries]
    iterations = [float(info.get("search_iterations", 0)) for _, info in entries]
    statuses = [str(info.get("status", "unknown")) for _, info in entries]
    colors = [PALETTE["Baseline"]] + [PALETTE["Age-Specialist"]] * len(AGE_GROUPS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar([label for label, _ in entries], scores, color=colors)
    axes[0].set_title("Best CV ROC-AUC During Tuning")
    axes[0].set_ylabel("CV ROC-AUC")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar([label for label, _ in entries], iterations, color=colors)
    axes[1].set_title("Hyperparameter Search Iterations")
    axes[1].set_ylabel("Iterations")
    axes[1].tick_params(axis="x", rotation=20)

    for index, status in enumerate(statuses):
        axes[1].text(index, iterations[index] + 0.1, status, ha="center", va="bottom", rotation=20, fontsize=8)

    fig.suptitle(f"{model_label}: Hyperparameter Tuning Summary", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_model_visualizations(
    model_key: str,
    model_label: str,
    baseline_overall: Dict[str, Any],
    specialist_overall: Dict[str, Any],
    baseline_group_metrics: Dict[str, Dict[str, Any]],
    specialist_group_metrics: Dict[str, Dict[str, Any]],
    baseline_sampling: Dict[str, Any],
    specialist_sampling: Dict[str, Dict[str, Any]],
    baseline_feature_info: Dict[str, Any],
    specialist_feature_selection: Dict[str, Dict[str, Any] | None],
    baseline_model_info: Dict[str, Any],
    specialist_model_summary: Dict[str, Dict[str, Any]],
) -> List[Path]:
    output_dir = _model_dir(model_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    return [
        plot_overall_metrics(
            output_dir / "overall_metrics_comparison.png",
            baseline_overall,
            specialist_overall,
            model_label,
        ),
        plot_per_cohort_metrics(
            output_dir / "per_cohort_metrics_comparison.png",
            baseline_group_metrics,
            specialist_group_metrics,
            model_label,
        ),
        plot_confusion_matrices(
            output_dir / "confusion_matrix_comparison.png",
            baseline_overall,
            specialist_overall,
            model_label,
        ),
        plot_sampling_summary(
            output_dir / "sampling_comparison.png",
            baseline_sampling,
            specialist_sampling,
            model_label,
        ),
        plot_feature_selection(
            output_dir / "feature_selection_comparison.png",
            baseline_feature_info,
            specialist_feature_selection,
            model_label,
        ),
        plot_tuning_summary(
            output_dir / "hyperparameter_tuning_summary.png",
            baseline_model_info,
            specialist_model_summary,
            model_label,
        ),
    ]


def plot_cross_model_overall_metrics(
    output_path: Path,
    model_results: Dict[str, Dict[str, Any]],
) -> Path:
    _style()
    model_labels = [result["model_name"] for result in model_results.values()]
    x = np.arange(len(model_labels))
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()
    summary_metrics = [
        ("accuracy", "Accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("f1", "F1"),
    ]

    for ax, (metric_key, metric_label) in zip(axes, summary_metrics):
        baseline_values = [
            _safe_float(result["baseline"]["overall"].get(metric_key))
            for result in model_results.values()
        ]
        specialist_values = [
            _safe_float(result["age_specialist"]["overall"].get(metric_key))
            for result in model_results.values()
        ]
        ax.bar(x - width / 2, baseline_values, width, label="Baseline", color=PALETTE["Baseline"])
        ax.bar(x + width / 2, specialist_values, width, label="Age-Specialist", color=PALETTE["Age-Specialist"])
        ax.set_title(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Cross-Model Holdout Performance Summary", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_cross_model_metric_delta(
    output_path: Path,
    model_results: Dict[str, Dict[str, Any]],
) -> Path:
    _style()
    metric_keys = [metric_key for metric_key, _ in OVERALL_METRICS]
    metric_labels = [label for _, label in OVERALL_METRICS]
    model_labels = [result["model_name"] for result in model_results.values()]

    delta_matrix = []
    for result in model_results.values():
        row = []
        for metric_key in metric_keys:
            baseline_value = _safe_float(result["baseline"]["overall"].get(metric_key))
            specialist_value = _safe_float(result["age_specialist"]["overall"].get(metric_key))
            row.append(specialist_value - baseline_value)
        delta_matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 0.7 * max(4, len(model_labels)) + 2))
    sns.heatmap(
        np.asarray(delta_matrix, dtype=float),
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0.0,
        xticklabels=metric_labels,
        yticklabels=model_labels,
        ax=ax,
    )
    ax.set_title("Age-Specialist Minus Baseline Holdout Metric Delta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_all_classifiers_comparison(
    output_path: Path,
    model_results: Dict[str, Dict[str, Any]],
) -> Path:
    _style()
    ranking_metric = "roc_auc"
    ordered_results = sorted(
        model_results.values(),
        key=lambda result: max(
            _safe_float(result["baseline"]["overall"].get(ranking_metric)),
            _safe_float(result["age_specialist"]["overall"].get(ranking_metric)),
        ),
        reverse=True,
    )
    model_labels = [result["model_name"] for result in ordered_results]
    metric_specs = [
        ("accuracy", "Accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("f1", "F1"),
    ]

    fig, axes = plt.subplots(1, len(metric_specs), figsize=(22, 0.65 * max(6, len(model_labels)) + 2), sharey=True)
    if len(metric_specs) == 1:
        axes = [axes]

    y = np.arange(len(model_labels))
    bar_height = 0.36

    for ax, (metric_key, metric_label) in zip(axes, metric_specs):
        baseline_values = [
            _safe_float(result["baseline"]["overall"].get(metric_key))
            for result in ordered_results
        ]
        specialist_values = [
            _safe_float(result["age_specialist"]["overall"].get(metric_key))
            for result in ordered_results
        ]
        ax.barh(y - bar_height / 2, baseline_values, height=bar_height, color=PALETTE["Baseline"], label="Baseline")
        ax.barh(
            y + bar_height / 2,
            specialist_values,
            height=bar_height,
            color=PALETTE["Age-Specialist"],
            label="Age-Specialist",
        )
        ax.set_title(metric_label)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Score")
        ax.set_yticks(y)
        ax.set_yticklabels(model_labels)
        ax.invert_yaxis()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("All Classifiers Comparison Across Key Holdout Metrics", y=0.99, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_cross_model_visualizations(model_results: Dict[str, Dict[str, Any]]) -> List[Path]:
    if not model_results:
        return []

    output_dir = _summary_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_cross_model_overall_metrics(
            output_dir / "cross_model_overall_metrics.png",
            model_results,
        ),
        plot_cross_model_metric_delta(
            output_dir / "cross_model_metric_delta_heatmap.png",
            model_results,
        ),
        plot_all_classifiers_comparison(
            output_dir / "all_classifiers_comparison.png",
            model_results,
        ),
    ]
