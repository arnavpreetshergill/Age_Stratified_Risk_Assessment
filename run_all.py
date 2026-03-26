from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from project_paths import (
    CORRELATION_HEATMAP_FILE,
    DATASETS_DIR,
    EXPLORATORY_VISUALIZATIONS_DIR,
    PERFORMANCE_VISUALIZATIONS_DIR,
    SHAP_VISUALIZATIONS_DIR,
    VISUALIZATIONS_DIR,
    ensure_root_artifact_dirs,
)

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("Preprocess raw data", ROOT / "preprocess.py"),
    ("Augment processed training data", ROOT / "dataGeneration.py"),
    ("Split augmented cohorts", ROOT / "data_split.py"),
    ("Run baseline vs age-specialist comparison", ROOT / "performance_comparison.py"),
    ("Generate SHAP explanations", ROOT / "gradient_boost_SHAP.py"),
    ("Generate age-group distribution plot", ROOT / "vis.py"),
    ("Generate correlation heatmap", ROOT / "correlation.py"),
    ("Run SMOTE subproject comparison", ROOT / "smote_comparison_project" / "compare_smote_vs_no_smote.py"),
]


def run_step(step_name: str, script_path: Path) -> None:
    print("=" * 88, flush=True)
    print(step_name, flush=True)
    print(f"Running: {script_path.relative_to(ROOT)}", flush=True)
    print("=" * 88, flush=True)
    subprocess.run([sys.executable, str(script_path)], cwd=ROOT, check=True)
    print(flush=True)


def main() -> None:
    ensure_root_artifact_dirs()
    for step_name, script_path in STEPS:
        run_step(step_name, script_path)

    print("=" * 88, flush=True)
    print("PIPELINE COMPLETE", flush=True)
    print("=" * 88, flush=True)
    print(f"Datasets: {DATASETS_DIR}", flush=True)
    print(f"Visualizations: {VISUALIZATIONS_DIR}", flush=True)
    print(f"Exploratory visuals: {EXPLORATORY_VISUALIZATIONS_DIR}", flush=True)
    print(f"Performance comparison visuals: {PERFORMANCE_VISUALIZATIONS_DIR}", flush=True)
    print(f"SHAP visuals: {SHAP_VISUALIZATIONS_DIR}", flush=True)
    print(f"Correlation heatmap: {CORRELATION_HEATMAP_FILE}", flush=True)
    print(f"SMOTE subproject results: {ROOT / 'smote_comparison_project' / 'results'}", flush=True)
    print(f"SMOTE subproject reports: {ROOT / 'smote_comparison_project' / 'reports'}", flush=True)
    print(f"SMOTE subproject visualizations: {ROOT / 'smote_comparison_project' / 'visualizations'}", flush=True)


if __name__ == "__main__":
    main()
