from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = ROOT_DIR / "datasets"
RAW_DATA_DIR = DATASETS_DIR / "raw"
PROCESSED_DATA_DIR = DATASETS_DIR / "processed"
COHORTS_DIR = DATASETS_DIR / "cohorts"
VISUALIZATIONS_DIR = ROOT_DIR / "visualizations"
EXPLORATORY_VISUALIZATIONS_DIR = VISUALIZATIONS_DIR / "exploratory"
PERFORMANCE_VISUALIZATIONS_DIR = VISUALIZATIONS_DIR / "performance"
SHAP_VISUALIZATIONS_DIR = VISUALIZATIONS_DIR / "shap"

RAW_DATA_FILE = RAW_DATA_DIR / "heart_statlog_cleveland_hungary_final(1).csv"

PROCESSED_FULL_FILE = PROCESSED_DATA_DIR / "processed_heart_data_full.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "processed_train.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "processed_test.csv"
PROCESSED_TRAIN_100K_FILE = PROCESSED_DATA_DIR / "processed_train_100k_stratified.csv"

YOUNG_COHORT_FILE = COHORTS_DIR / "heart_data_young.csv"
MIDDLE_COHORT_FILE = COHORTS_DIR / "heart_data_middle.csv"
ELDERLY_COHORT_FILE = COHORTS_DIR / "heart_data_elderly.csv"

AGE_GROUP_DISTRIBUTION_FILE = EXPLORATORY_VISUALIZATIONS_DIR / "age_group_distribution.png"
CORRELATION_HEATMAP_FILE = EXPLORATORY_VISUALIZATIONS_DIR / "correlation_heatmap.png"


def ensure_root_artifact_dirs() -> None:
    for path in [
        DATASETS_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        COHORTS_DIR,
        VISUALIZATIONS_DIR,
        EXPLORATORY_VISUALIZATIONS_DIR,
        PERFORMANCE_VISUALIZATIONS_DIR,
        SHAP_VISUALIZATIONS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
