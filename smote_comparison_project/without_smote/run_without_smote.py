from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_utils import run_variant


if __name__ == "__main__":
    data_file = ROOT / "heart_statlog_cleveland_hungary_final(1).csv"
    output_file = Path(__file__).resolve().parent / "results_without_smote.json"
    run_variant(
        raw_file=data_file,
        use_smote=False,
        output_json=output_file,
        variant_name="WITHOUT_SMOTE",
    )
