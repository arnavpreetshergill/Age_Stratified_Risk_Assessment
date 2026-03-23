from pathlib import Path

SUBPROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SUBPROJECT_ROOT.parent

RESULTS_DIR = SUBPROJECT_ROOT / "results"
WITH_SMOTE_RESULTS_DIR = RESULTS_DIR / "with_smote"
WITHOUT_SMOTE_RESULTS_DIR = RESULTS_DIR / "without_smote"
REPORTS_DIR = SUBPROJECT_ROOT / "reports"
VISUALIZATIONS_DIR = SUBPROJECT_ROOT / "visualizations"

ROOT_PROCESSED_TRAIN_FILE = PROJECT_ROOT / "datasets" / "processed" / "processed_train_100k_stratified.csv"
ROOT_PROCESSED_TEST_FILE = PROJECT_ROOT / "datasets" / "processed" / "processed_test.csv"
ROOT_RAW_DATA_FILE = PROJECT_ROOT / "datasets" / "raw" / "heart_statlog_cleveland_hungary_final(1).csv"

RESULTS_WITH_SMOTE_FILE = WITH_SMOTE_RESULTS_DIR / "results_with_smote.json"
RESULTS_WITHOUT_SMOTE_FILE = WITHOUT_SMOTE_RESULTS_DIR / "results_without_smote.json"
REPORT_FILE = REPORTS_DIR / "SMOTE_COMPARISON_REPORT.md"


def ensure_subproject_artifact_dirs() -> None:
    for path in [
        RESULTS_DIR,
        WITH_SMOTE_RESULTS_DIR,
        WITHOUT_SMOTE_RESULTS_DIR,
        REPORTS_DIR,
        VISUALIZATIONS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
