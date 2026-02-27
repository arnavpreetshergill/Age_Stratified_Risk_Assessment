from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "target"
NUMERIC_FEATURES = ["age", "resting_bp_s", "cholesterol", "max_heart_rate", "oldpeak"]
CATEGORICAL_FEATURES = ["chest_pain_type", "resting_ecg", "st_slope"]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_and_clean_data(input_file: Path) -> pd.DataFrame:
    df = pd.read_csv(input_file)
    df_clean = df.drop_duplicates().copy()
    df_clean.columns = [col.lower().replace(" ", "_") for col in df_clean.columns]
    df_clean = df_clean[df_clean["st_slope"] != 0].reset_index(drop=True)
    return df_clean


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
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


def build_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )


def class_counts(y: pd.Series) -> Dict[int, int]:
    counts = y.astype(int).value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }

    if np.unique(y_true_arr).size > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
        metrics["pr_auc"] = float(average_precision_score(y_true_arr, y_prob_arr))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    return metrics


def smote_resample_binary(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
    k_neighbors: int = 5,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    X_work = X_train.reset_index(drop=True).copy()
    y_work = y_train.reset_index(drop=True).astype(int)

    counts = y_work.value_counts()
    if counts.size < 2:
        return X_work, y_work, {
            "applied": False,
            "reason": "single_class_train_data",
            "before_counts": class_counts(y_work),
            "after_counts": class_counts(y_work),
            "generated_samples": 0,
            "k_neighbors_used": 0,
        }

    minority_label = int(counts.idxmin())
    majority_count = int(counts.max())
    minority_count = int(counts.min())
    samples_to_generate = majority_count - minority_count

    if samples_to_generate <= 0:
        return X_work, y_work, {
            "applied": False,
            "reason": "already_balanced",
            "before_counts": class_counts(y_work),
            "after_counts": class_counts(y_work),
            "generated_samples": 0,
            "k_neighbors_used": 0,
        }

    X_minority = X_work[y_work == minority_label].to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)

    if len(X_minority) < 2:
        sample_idx = rng.integers(0, len(X_minority), size=samples_to_generate)
        synthetic = X_minority[sample_idx]
        k_used = 0
    else:
        k_used = min(k_neighbors, len(X_minority) - 1)
        neighbors = NearestNeighbors(n_neighbors=k_used + 1)
        neighbors.fit(X_minority)
        nn_indices = neighbors.kneighbors(X_minority, return_distance=False)[:, 1:]

        synthetic = np.empty((samples_to_generate, X_minority.shape[1]), dtype=float)
        for i in range(samples_to_generate):
            idx = int(rng.integers(0, len(X_minority)))
            x_i = X_minority[idx]
            neighbor_pool = nn_indices[idx]
            n_idx = int(rng.choice(neighbor_pool)) if len(neighbor_pool) > 0 else idx
            x_n = X_minority[n_idx]
            gap = float(rng.random())
            synthetic[i] = x_i + gap * (x_n - x_i)

    synthetic_df = pd.DataFrame(synthetic, columns=X_work.columns)
    synthetic_target = pd.Series(
        np.full(samples_to_generate, minority_label, dtype=int),
        name=y_work.name,
    )

    X_balanced = pd.concat([X_work, synthetic_df], ignore_index=True)
    y_balanced = pd.concat([y_work, synthetic_target], ignore_index=True)

    return X_balanced, y_balanced, {
        "applied": True,
        "minority_label": minority_label,
        "before_counts": class_counts(y_work),
        "after_counts": class_counts(y_balanced),
        "generated_samples": int(samples_to_generate),
        "k_neighbors_used": int(k_used),
    }


def _to_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        float_value = float(value)
        return None if np.isnan(float_value) else float_value
    return value


def run_variant(
    raw_file: Path,
    use_smote: bool,
    output_json: Path | None,
    variant_name: str,
) -> Dict[str, Any]:
    df_clean = load_and_clean_data(raw_file)
    X_raw = df_clean.drop(columns=[TARGET_COL])
    y_raw = df_clean[TARGET_COL].astype(int)

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw,
        y_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_raw,
    )

    preprocessor = build_preprocessor()
    train_df = transform_with_preprocessor(X_train_raw, y_train_raw, preprocessor, fit=True)
    test_df = transform_with_preprocessor(X_test_raw, y_test_raw, preprocessor, fit=False)

    feature_cols = [col for col in train_df.columns if col != TARGET_COL]
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(int)

    if use_smote:
        X_train_final, y_train_final, sampling_info = smote_resample_binary(X_train, y_train)
    else:
        X_train_final = X_train.reset_index(drop=True)
        y_train_final = y_train.reset_index(drop=True)
        sampling_info = {
            "applied": False,
            "reason": "smote_disabled",
            "before_counts": class_counts(y_train),
            "after_counts": class_counts(y_train),
            "generated_samples": 0,
            "k_neighbors_used": 0,
        }

    model = build_model()
    model.fit(X_train_final, y_train_final)

    y_pred = model.predict(X_test).astype(int)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)

    results: Dict[str, Any] = {
        "variant": variant_name,
        "use_smote": bool(use_smote),
        "dataset_rows_after_cleaning": int(len(df_clean)),
        "target_definition": "1 = cardiovascular disease, 0 = no cardiovascular disease",
        "train_rows_before_sampling": int(len(y_train)),
        "train_rows_after_sampling": int(len(y_train_final)),
        "test_rows": int(len(y_test)),
        "sampling": sampling_info,
        "metrics": metrics,
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(_to_json_ready(results), indent=2),
            encoding="utf-8",
        )

    print(f"[{variant_name}] Train before sampling: {class_counts(y_train)}")
    print(f"[{variant_name}] Train after sampling:  {class_counts(y_train_final)}")
    roc_auc_text = "N/A" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.4f}"
    print(
        f"[{variant_name}] Accuracy={metrics['accuracy']:.4f}, "
        f"Recall={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}, "
        f"ROC-AUC={roc_auc_text}"
    )

    return results
