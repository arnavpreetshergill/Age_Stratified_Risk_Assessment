from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import (
    RESULTS_WITH_SMOTE_FILE,
    ROOT_PROCESSED_TEST_FILE,
    ROOT_PROCESSED_TRAIN_FILE,
    ROOT_RAW_DATA_FILE,
    ensure_subproject_artifact_dirs,
)
from shared_utils import run_variant_on_generated_data


if __name__ == "__main__":
    ensure_subproject_artifact_dirs()
    run_variant_on_generated_data(
        processed_train_file=ROOT_PROCESSED_TRAIN_FILE,
        processed_test_file=ROOT_PROCESSED_TEST_FILE,
        raw_reference_file=ROOT_RAW_DATA_FILE,
        use_smote=True,
        output_json=RESULTS_WITH_SMOTE_FILE,
        variant_name="WITH_SMOTE",
    )
