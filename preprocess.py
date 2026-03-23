import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project_paths import (
    PROCESSED_FULL_FILE,
    PROCESSED_TEST_FILE,
    PROCESSED_TRAIN_FILE,
    RAW_DATA_FILE,
    ensure_root_artifact_dirs,
)

TARGET_COL = "target"
NUMERIC_FEATURES = ["age", "resting_bp_s", "cholesterol", "max_heart_rate", "oldpeak"]
CATEGORICAL_FEATURES = ["chest_pain_type", "resting_ecg", "st_slope"]


def load_and_clean_data(input_file):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")

    df_clean = df.drop_duplicates().copy()
    df_clean.columns = [col.lower().replace(" ", "_") for col in df_clean.columns]
    df_clean = df_clean[df_clean["st_slope"] != 0].reset_index(drop=True)

    print(f"Cleaned shape: {df_clean.shape}")
    return df_clean


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )


def transform_with_preprocessor(X, y, preprocessor, fit=False):
    X_processed = preprocessor.fit_transform(X) if fit else preprocessor.transform(X)
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    passthrough_cols = [
        col for col in X.columns if col not in NUMERIC_FEATURES and col not in CATEGORICAL_FEATURES
    ]
    all_feature_names = NUMERIC_FEATURES + list(cat_names) + passthrough_cols

    out_df = pd.DataFrame(X_processed, columns=all_feature_names)
    out_df[TARGET_COL] = y.reset_index(drop=True)
    return out_df


def preprocess_full_dataset(input_file, output_file):
    df_clean = load_and_clean_data(input_file)
    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]

    preprocessor = build_preprocessor()
    final_df = transform_with_preprocessor(X, y, preprocessor, fit=True)
    final_df.to_csv(output_file, index=False)

    print(f"Saved processed full dataset: {output_file} ({final_df.shape})")
    return final_df


def preprocess_train_test_split(
    input_file,
    train_output_file="processed_train.csv",
    test_output_file="processed_test.csv",
    test_size=0.2,
    random_state=42,
):
    df_clean = load_and_clean_data(input_file)
    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    train_df = transform_with_preprocessor(X_train, y_train, preprocessor, fit=True)
    test_df = transform_with_preprocessor(X_test, y_test, preprocessor, fit=False)

    train_df.to_csv(train_output_file, index=False)
    test_df.to_csv(test_output_file, index=False)

    print(f"Saved train split: {train_output_file} ({train_df.shape})")
    print(f"Saved test split: {test_output_file} ({test_df.shape})")
    return train_df, test_df


if __name__ == "__main__":
    ensure_root_artifact_dirs()
    preprocess_full_dataset(
        RAW_DATA_FILE,
        PROCESSED_FULL_FILE,
    )
    preprocess_train_test_split(
        RAW_DATA_FILE,
        PROCESSED_TRAIN_FILE,
        PROCESSED_TEST_FILE,
    )
