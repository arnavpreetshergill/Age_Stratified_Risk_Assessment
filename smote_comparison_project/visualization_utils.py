from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from project_paths import VISUALIZATIONS_DIR

AGE_GROUPS = ["Young", "Middle", "Elderly"]
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
PALETTE = {
    "Without SMOTE": "#3A5A78",
    "With SMOTE": "#D97A4A",
}


def _safe_float(value: Any) -> float:
    if value is None:
        return np.nan
    return float(value)


def _count_value(counts: Dict[Any, Any], label: int) -> int:
    return int(counts.get(label, counts.get(str(label), 0)))


def _style() -> None:
    sns.set_theme(style="whitegrid")


def plot_overall_metrics(
    output_path: Path,
    without_smote: Dict[str, Any],
    with_smote: Dict[str, Any],
) -> Path:
    _style()
    metric_labels = [label for _, label in OVERALL_METRICS]
    without_values = [
        _safe_float(without_smote["overall_metrics"].get(metric_key))
        for metric_key, _ in OVERALL_METRICS
    ]
    with_values = [
        _safe_float(with_smote["overall_metrics"].get(metric_key))
        for metric_key, _ in OVERALL_METRICS
    ]

    x = np.arange(len(metric_labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, without_values, width, label="Without SMOTE", color=PALETTE["Without SMOTE"])
    ax.bar(x + width / 2, with_values, width, label="With SMOTE", color=PALETTE["With SMOTE"])
    ax.set_title("Holdout Test Overall Metric Comparison")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_cohort_metrics(
    output_path: Path,
    without_smote: Dict[str, Any],
    with_smote: Dict[str, Any],
) -> Path:
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    x = np.arange(len(AGE_GROUPS))
    width = 0.34

    for index, (metric_key, metric_label) in enumerate(COHORT_METRICS):
        ax = axes[index]
        without_values = [
            _safe_float(without_smote["cohorts"][cohort]["metrics"].get(metric_key))
            for cohort in AGE_GROUPS
        ]
        with_values = [
            _safe_float(with_smote["cohorts"][cohort]["metrics"].get(metric_key))
            for cohort in AGE_GROUPS
        ]
        ax.bar(
            x - width / 2,
            without_values,
            width,
            label="Without SMOTE",
            color=PALETTE["Without SMOTE"],
        )
        ax.bar(
            x + width / 2,
            with_values,
            width,
            label="With SMOTE",
            color=PALETTE["With SMOTE"],
        )
        ax.set_title(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(AGE_GROUPS)
        ax.set_ylim(0, 1.05)

    axes[len(COHORT_METRICS)].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Holdout Test Per-Cohort Metric Comparison", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrices(
    output_path: Path,
    without_smote: Dict[str, Any],
    with_smote: Dict[str, Any],
) -> Path:
    _style()
    without_cm = np.array(
        [
            [without_smote["overall_metrics"]["tn"], without_smote["overall_metrics"]["fp"]],
            [without_smote["overall_metrics"]["fn"], without_smote["overall_metrics"]["tp"]],
        ]
    )
    with_cm = np.array(
        [
            [with_smote["overall_metrics"]["tn"], with_smote["overall_metrics"]["fp"]],
            [with_smote["overall_metrics"]["fn"], with_smote["overall_metrics"]["tp"]],
        ]
    )

    vmax = max(without_cm.max(), with_cm.max())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    sns.heatmap(
        without_cm,
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
    axes[0].set_title("Without SMOTE")
    sns.heatmap(
        with_cm,
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
    axes[1].set_title("With SMOTE")
    fig.suptitle("Holdout Test Overall Confusion Matrix Comparison", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sampling_summary(
    output_path: Path,
    without_smote: Dict[str, Any],
    with_smote: Dict[str, Any],
) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    count_categories = ["Without Before", "Without After", "With Before", "With After"]
    class_0 = [
        _count_value(without_smote["train_class_counts_before_sampling_overall"], 0),
        _count_value(without_smote["train_class_counts_after_sampling_overall"], 0),
        _count_value(with_smote["train_class_counts_before_sampling_overall"], 0),
        _count_value(with_smote["train_class_counts_after_sampling_overall"], 0),
    ]
    class_1 = [
        _count_value(without_smote["train_class_counts_before_sampling_overall"], 1),
        _count_value(without_smote["train_class_counts_after_sampling_overall"], 1),
        _count_value(with_smote["train_class_counts_before_sampling_overall"], 1),
        _count_value(with_smote["train_class_counts_after_sampling_overall"], 1),
    ]

    x_counts = np.arange(len(count_categories))
    width = 0.34
    axes[0].bar(x_counts - width / 2, class_0, width, label="Class 0", color="#5B7C99")
    axes[0].bar(x_counts + width / 2, class_1, width, label="Class 1", color="#C86A43")
    axes[0].set_title("Train Class Counts")
    axes[0].set_xticks(x_counts)
    axes[0].set_xticklabels(count_categories, rotation=20, ha="right")
    axes[0].set_ylabel("Rows")
    axes[0].legend()

    sample_categories = AGE_GROUPS + ["Overall"]
    without_generated = [0, 0, 0, 0]
    with_generated = [
        int(with_smote["cohorts"][cohort]["sampling"]["generated_samples"])
        for cohort in AGE_GROUPS
    ]
    with_generated.append(
        int(with_smote["train_rows_after_sampling"] - with_smote["train_rows_before_sampling"])
    )

    x_samples = np.arange(len(sample_categories))
    axes[1].bar(
        x_samples - width / 2,
        without_generated,
        width,
        label="Without SMOTE",
        color=PALETTE["Without SMOTE"],
    )
    axes[1].bar(
        x_samples + width / 2,
        with_generated,
        width,
        label="With SMOTE",
        color=PALETTE["With SMOTE"],
    )
    axes[1].set_title("Generated Samples Added by SMOTE")
    axes[1].set_xticks(x_samples)
    axes[1].set_xticklabels(sample_categories)
    axes[1].set_ylabel("Generated rows")
    axes[1].legend()

    fig.suptitle("Holdout Test Sampling Comparison", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_selection(
    output_path: Path,
    without_smote: Dict[str, Any],
    with_smote: Dict[str, Any],
) -> Path:
    _style()
    fig, axes = plt.subplots(len(AGE_GROUPS), 2, figsize=(14, 11))

    for row_index, cohort in enumerate(AGE_GROUPS):
        for column_index, (title, payload, color) in enumerate(
            [
                ("Without SMOTE", without_smote, PALETTE["Without SMOTE"]),
                ("With SMOTE", with_smote, PALETTE["With SMOTE"]),
            ]
        ):
            ax = axes[row_index, column_index]
            feature_info = payload["cohorts"][cohort].get("feature_selection", {})
            importances = feature_info.get("selected_feature_importances")
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
            ax.set_title(f"{cohort} - {title}")
            ax.set_xlabel("Importance")

    fig.suptitle("Selected Feature Importance by Cohort", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_visualizations(
    without_smote: Dict[str, Any],
    with_smote: Dict[str, Any],
) -> List[Path]:
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    outputs = [
        plot_overall_metrics(VISUALIZATIONS_DIR / "overall_metrics_comparison.png", without_smote, with_smote),
        plot_cohort_metrics(
            VISUALIZATIONS_DIR / "per_cohort_metrics_comparison.png",
            without_smote,
            with_smote,
        ),
        plot_confusion_matrices(
            VISUALIZATIONS_DIR / "confusion_matrix_comparison.png",
            without_smote,
            with_smote,
        ),
        plot_sampling_summary(VISUALIZATIONS_DIR / "sampling_comparison.png", without_smote, with_smote),
        plot_feature_selection(
            VISUALIZATIONS_DIR / "feature_selection_comparison.png",
            without_smote,
            with_smote,
        ),
    ]
    return outputs
