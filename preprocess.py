from data_pipeline_utils import DEFAULT_TEST_SIZE, prepare_train_test_data
from project_paths import (
    PROCESSED_TEST_FILE,
    PROCESSED_TRAIN_FILE,
    RAW_DATA_FILE,
    ensure_root_artifact_dirs,
)


def preprocess_train_test_split(
    input_file,
    train_output_file="processed_train.csv",
    test_output_file="processed_test.csv",
    test_size=DEFAULT_TEST_SIZE,
    random_state=42,
):
    split_data = prepare_train_test_data(
        input_file,
        test_size=test_size,
        random_state=random_state,
    )
    train_df = split_data["train_df"]
    test_df = split_data["test_df"]

    train_df.to_csv(train_output_file, index=False)
    test_df.to_csv(test_output_file, index=False)

    print(f"Saved train split: {train_output_file} ({train_df.shape})")
    print(f"Saved test split: {test_output_file} ({test_df.shape})")
    return train_df, test_df


if __name__ == "__main__":
    ensure_root_artifact_dirs()
    preprocess_train_test_split(
        RAW_DATA_FILE,
        PROCESSED_TRAIN_FILE,
        PROCESSED_TEST_FILE,
    )
