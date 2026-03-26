from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "target"
AGE_COL = "age"
NUMERIC_FEATURES = ["age", "resting_bp_s", "cholesterol", "max_heart_rate", "oldpeak"]
CATEGORICAL_FEATURES = ["chest_pain_type", "resting_ecg", "st_slope"]
CATEGORICAL_CATEGORIES: list[list[int]] = [
    [1, 2, 3, 4],
    [0, 1, 2],
    [1, 2, 3],
]
AGE_GROUPS = ["Young", "Middle", "Elderly"]
COHORT_RULES = {
    "Young": "age < 45",
    "Middle": "45 <= age <= 65",
    "Elderly": "age > 65",
}


def load_and_clean_data(input_file: str | Any) -> pd.DataFrame:
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")

    df_clean = df.drop_duplicates().copy()
    df_clean.columns = [col.lower().replace(" ", "_") for col in df_clean.columns]
    df_clean = df_clean[df_clean["st_slope"] != 0].reset_index(drop=True)

    print(f"Cleaned shape: {df_clean.shape}")
    return df_clean


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    categories=CATEGORICAL_CATEGORIES,
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",
    )


def transform_with_preprocessor(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    fit: bool = False,
) -> pd.DataFrame:
    X_processed = preprocessor.fit_transform(X) if fit else preprocessor.transform(X)
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    passthrough_cols = [
        col for col in X.columns if col not in NUMERIC_FEATURES and col not in CATEGORICAL_FEATURES
    ]
    all_feature_names = NUMERIC_FEATURES + list(cat_names) + passthrough_cols

    out_df = pd.DataFrame(X_processed, columns=all_feature_names)
    out_df[TARGET_COL] = y.reset_index(drop=True).astype(int)
    return out_df


def split_raw_train_test(
    input_file: str | Any,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df_clean = load_and_clean_data(input_file)
    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def prepare_train_test_data(
    input_file: str | Any,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    X_train_raw, X_test_raw, y_train, y_test = split_raw_train_test(
        input_file,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor()
    train_df = transform_with_preprocessor(X_train_raw, y_train, preprocessor, fit=True)
    test_df = transform_with_preprocessor(X_test_raw, y_test, preprocessor, fit=False)
    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
        "preprocessor": preprocessor,
    }


def get_processed_age_cutoffs(preprocessor: ColumnTransformer) -> tuple[float, float]:
    scaler = preprocessor.named_transformers_["num"]
    age_idx = NUMERIC_FEATURES.index(AGE_COL)
    age_mean = float(scaler.mean_[age_idx])
    age_std = float(scaler.scale_[age_idx])
    return float((45 - age_mean) / age_std), float((65 - age_mean) / age_std)


def compute_age_z_thresholds(
    raw_file: str | Any,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[float, float, float, float]:
    split_data = prepare_train_test_data(
        raw_file,
        test_size=test_size,
        random_state=random_state,
    )
    preprocessor = split_data["preprocessor"]
    scaler = preprocessor.named_transformers_["num"]
    age_idx = NUMERIC_FEATURES.index(AGE_COL)
    age_z_45, age_z_65 = get_processed_age_cutoffs(preprocessor)
    raw_mean = float(scaler.mean_[age_idx])
    raw_std = float(scaler.scale_[age_idx])
    return age_z_45, age_z_65, raw_mean, raw_std


def assign_age_group_raw(age_series: pd.Series) -> np.ndarray:
    conditions = [
        age_series < 45,
        (age_series >= 45) & (age_series <= 65),
        age_series > 65,
    ]
    return np.select(conditions, AGE_GROUPS, default="Unknown")


def assign_age_group_processed(
    age_series: pd.Series,
    young_upper_bound: float,
    middle_upper_bound: float,
) -> pd.Series:
    conditions = [
        age_series < young_upper_bound,
        (age_series >= young_upper_bound) & (age_series <= middle_upper_bound),
        age_series > middle_upper_bound,
    ]
    return pd.Series(np.select(conditions, AGE_GROUPS, default="Unknown"), index=age_series.index)


def assert_frames_match(
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    artifact_name: str,
    atol: float = 1e-9,
) -> None:
    if list(candidate_df.columns) != list(reference_df.columns):
        raise ValueError(f"{artifact_name} columns do not match the canonical pipeline output.")
    if candidate_df.shape != reference_df.shape:
        raise ValueError(
            f"{artifact_name} shape does not match the canonical pipeline output: "
            f"expected {reference_df.shape}, got {candidate_df.shape}."
        )

    for column in reference_df.columns:
        reference_col = reference_df[column]
        candidate_col = candidate_df[column]
        if pd.api.types.is_numeric_dtype(reference_col) and pd.api.types.is_numeric_dtype(candidate_col):
            if not np.allclose(
                reference_col.to_numpy(dtype=float),
                candidate_col.to_numpy(dtype=float),
                atol=atol,
                rtol=0.0,
                equal_nan=True,
            ):
                raise ValueError(f"{artifact_name} values differ from the canonical pipeline output in '{column}'.")
            continue

        if not reference_col.equals(candidate_col):
            raise ValueError(f"{artifact_name} values differ from the canonical pipeline output in '{column}'.")
