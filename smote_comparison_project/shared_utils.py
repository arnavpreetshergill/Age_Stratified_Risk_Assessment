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
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "target"
AGE_COL = "age"
NUMERIC_FEATURES = ["age", "resting_bp_s", "cholesterol", "max_heart_rate", "oldpeak"]
CATEGORICAL_FEATURES = ["chest_pain_type", "resting_ecg", "st_slope"]
PROCESSED_FEATURES = [
    "age",
    "resting_bp_s",
    "cholesterol",
    "max_heart_rate",
    "oldpeak",
    "chest_pain_type_2",
    "chest_pain_type_3",
    "chest_pain_type_4",
    "resting_ecg_1",
    "resting_ecg_2",
    "st_slope_2",
    "st_slope_3",
    "sex",
    "fasting_blood_sugar",
    "exercise_angina",
]
AGE_GROUPS = ["Young", "Middle", "Elderly"]
COHORT_RULES = {
    "Young": "age < 45",
    "Middle": "45 <= age <= 65",
    "Elderly": "age > 65",
}
TEST_SIZE = 0.2
RANDOM_STATE = 42
GRIDSEARCH_SCORING = "f1"
FEATURE_SELECTION_TOP_K = 10
GRIDSEARCH_PARAM_GRID = {
    "n_estimators": [80, 140],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.15],
    "subsample": [0.8, 1.0],
}
BINARY_FEATURES = {"sex", "fasting_blood_sugar", "exercise_angina"}
ONE_HOT_GROUPS = [
    ["chest_pain_type_2", "chest_pain_type_3", "chest_pain_type_4"],
    ["resting_ecg_1", "resting_ecg_2"],
    ["st_slope_2", "st_slope_3"],
]


def get_age_group(
    age_value: float,
    young_upper_bound: float = 45.0,
    middle_upper_bound: float = 65.0,
) -> str:
    if age_value < young_upper_bound:
        return "Young"
    if age_value <= middle_upper_bound:
        return "Middle"
    return "Elderly"


def assign_age_groups(
    age_series: pd.Series,
    young_upper_bound: float = 45.0,
    middle_upper_bound: float = 65.0,
) -> pd.Series:
    return age_series.apply(
        lambda age_value: get_age_group(
            age_value,
            young_upper_bound=young_upper_bound,
            middle_upper_bound=middle_upper_bound,
        )
    )


def load_and_clean_data(input_file: Path) -> pd.DataFrame:
    df = pd.read_csv(input_file)
    df_clean = df.drop_duplicates().copy()
    df_clean.columns = [col.lower().replace(" ", "_") for col in df_clean.columns]
    df_clean = df_clean[df_clean["st_slope"] != 0].reset_index(drop=True)
    return df_clean


def load_processed_data(input_file: Path) -> pd.DataFrame:
    df = pd.read_csv(input_file)
    missing_cols = sorted(set(PROCESSED_FEATURES + [TARGET_COL]).difference(df.columns))
    if missing_cols:
        raise ValueError(
            f"Processed dataset '{input_file}' is missing required columns: {missing_cols}"
        )
    return df.copy()


def compute_processed_age_cutoffs(raw_reference_file: Path) -> Dict[str, float]:
    df_clean = load_and_clean_data(raw_reference_file)
    X_raw = df_clean.drop(columns=[TARGET_COL])
    y_raw = df_clean[TARGET_COL].astype(int)

    X_train_raw, _, _, _ = train_test_split(
        X_raw,
        y_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_raw,
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train_raw)

    scaler = preprocessor.named_transformers_["num"]
    age_idx = NUMERIC_FEATURES.index(AGE_COL)
    age_mean = float(scaler.mean_[age_idx])
    age_std = float(scaler.scale_[age_idx])

    return {
        "young_upper_bound_processed": float((45 - age_mean) / age_std),
        "middle_upper_bound_processed": float((65 - age_mean) / age_std),
        "raw_train_age_mean": age_mean,
        "raw_train_age_std": age_std,
    }


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


def build_default_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=1,
    )


def feature_selection_not_applied(
    reason: str,
    feature_names: pd.Index | list[str],
) -> Dict[str, Any]:
    selected_features = list(feature_names)
    return {
        "applied": False,
        "reason": reason,
        "method": "xgboost_feature_importance_top_k",
        "requested_top_k": int(FEATURE_SELECTION_TOP_K),
        "candidate_feature_count": int(len(selected_features)),
        "selected_feature_count": int(len(selected_features)),
        "selected_features": selected_features,
        "selected_feature_importances": None,
    }


def select_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    top_k: int = FEATURE_SELECTION_TOP_K,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    candidate_features = list(X_train.columns)
    if len(candidate_features) <= top_k:
        info = feature_selection_not_applied(
            reason="feature_count_within_limit",
            feature_names=candidate_features,
        )
        return X_train.copy(), X_test.loc[:, candidate_features].copy(), info

    if y_train.nunique() < 2:
        info = feature_selection_not_applied(
            reason="single_class_train_data",
            feature_names=candidate_features,
        )
        return X_train.copy(), X_test.loc[:, candidate_features].copy(), info

    selector_model = build_default_model()
    selector_model.fit(X_train, y_train.astype(int))
    importances = pd.Series(
        selector_model.feature_importances_,
        index=X_train.columns,
        dtype=float,
    ).sort_values(ascending=False, kind="stable")

    selected_features = list(importances.head(top_k).index)
    feature_importances = {
        feature_name: float(importances.loc[feature_name])
        for feature_name in selected_features
    }
    info = {
        "applied": True,
        "reason": "top_k_selected",
        "method": "xgboost_feature_importance_top_k",
        "requested_top_k": int(top_k),
        "candidate_feature_count": int(len(candidate_features)),
        "selected_feature_count": int(len(selected_features)),
        "selected_features": selected_features,
        "selected_feature_importances": feature_importances,
    }
    return (
        X_train.loc[:, selected_features].copy(),
        X_test.loc[:, selected_features].copy(),
        info,
    )


def class_counts(y: pd.Series) -> Dict[int, int]:
    counts = y.astype(int).value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}


def _mixed_feature_indices(X: pd.DataFrame) -> Tuple[list[int], list[int], list[list[int]]]:
    feature_names = list(X.columns)
    feature_index = {name: idx for idx, name in enumerate(feature_names)}

    grouped_feature_names: set[str] = set()
    grouped_indices: list[list[int]] = []
    for group in ONE_HOT_GROUPS:
        present = [feature_index[name] for name in group if name in feature_index]
        if present:
            grouped_indices.append(present)
            grouped_feature_names.update(name for name in group if name in feature_index)

    binary_indices: list[int] = []
    for idx, feature_name in enumerate(feature_names):
        if feature_name in grouped_feature_names:
            continue
        if feature_name in BINARY_FEATURES:
            binary_indices.append(idx)
            continue
        values = X.iloc[:, idx].dropna()
        if not values.empty and values.isin([0, 1]).all():
            binary_indices.append(idx)

    discrete_indices = set(binary_indices)
    for group in grouped_indices:
        discrete_indices.update(group)
    continuous_indices = [idx for idx in range(len(feature_names)) if idx not in discrete_indices]
    return continuous_indices, binary_indices, grouped_indices


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
        continuous_indices, binary_indices, grouped_indices = _mixed_feature_indices(X_work)
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
            if continuous_indices:
                synthetic[i, continuous_indices] = (
                    x_i[continuous_indices] + gap * (x_n[continuous_indices] - x_i[continuous_indices])
                )
            for binary_idx in binary_indices:
                synthetic[i, binary_idx] = x_i[binary_idx] if rng.random() < 0.5 else x_n[binary_idx]
            for group_indices in grouped_indices:
                donor = x_i if rng.random() < 0.5 else x_n
                synthetic[i, group_indices] = donor[group_indices]

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


def tune_xgboost_with_gridsearch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    y_work = y_train.astype(int).reset_index(drop=True)
    counts = y_work.value_counts()
    min_class_count = int(counts.min()) if counts.size > 0 else 0

    if counts.size < 2 or min_class_count < 2:
        model = build_default_model()
        model.fit(X_train, y_work)
        return model, {
            "method": "GridSearchCV",
            "status": "fallback_default_model",
            "reason": "insufficient_class_diversity_for_cv",
            "cv_folds": 0,
            "param_grid_size": len(list(ParameterGrid(GRIDSEARCH_PARAM_GRID))),
            "best_params": {
                "n_estimators": int(model.get_params()["n_estimators"]),
                "max_depth": int(model.get_params()["max_depth"]),
                "learning_rate": float(model.get_params()["learning_rate"]),
                "subsample": float(model.get_params()["subsample"]),
            },
            "best_cv_score_f1": None,
        }

    cv_folds = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=build_default_model(),
        param_grid=GRIDSEARCH_PARAM_GRID,
        scoring=GRIDSEARCH_SCORING,
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_work)

    best_model = search.best_estimator_
    return best_model, {
        "method": "GridSearchCV",
        "status": "tuned",
        "scoring": GRIDSEARCH_SCORING,
        "cv_folds": int(cv_folds),
        "param_grid_size": len(list(ParameterGrid(GRIDSEARCH_PARAM_GRID))),
        "best_params": search.best_params_,
        "best_cv_score_f1": float(search.best_score_),
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


def save_results(output_json: Path | None, results: Dict[str, Any]) -> None:
    if output_json is None:
        return

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(_to_json_ready(results), indent=2),
        encoding="utf-8",
    )


def print_variant_summary(results: Dict[str, Any]) -> None:
    variant_name = results["variant"]
    overall_before_counts = results["train_class_counts_before_sampling_overall"]
    overall_after_counts = results["train_class_counts_after_sampling_overall"]

    print(f"[{variant_name}] Train before sampling (overall): {overall_before_counts}")
    print(f"[{variant_name}] Train after sampling (overall):  {overall_after_counts}")
    for cohort in AGE_GROUPS:
        cohort_data = results["cohorts"].get(cohort, {})
        cohort_metrics = cohort_data.get("metrics")
        tuning = cohort_data.get("tuning", {})
        if cohort_metrics is None:
            print(f"[{variant_name}] {cohort}: metrics unavailable")
            continue
        best_params = tuning.get("best_params")
        roc_auc_text = "N/A" if cohort_metrics["roc_auc"] is None else f"{cohort_metrics['roc_auc']:.4f}"
        print(
            f"[{variant_name}] {cohort}: "
            f"Acc={cohort_metrics['accuracy']:.4f}, "
            f"Recall={cohort_metrics['recall']:.4f}, "
            f"F1={cohort_metrics['f1']:.4f}, "
            f"ROC-AUC={roc_auc_text}, "
            f"BestParams={best_params}"
        )

    overall_metrics = results["overall_metrics"]
    overall_roc_auc_text = (
        "N/A" if overall_metrics["roc_auc"] is None else f"{overall_metrics['roc_auc']:.4f}"
    )
    print(
        f"[{variant_name}] OVERALL: "
        f"Accuracy={overall_metrics['accuracy']:.4f}, "
        f"Recall={overall_metrics['recall']:.4f}, "
        f"F1={overall_metrics['f1']:.4f}, "
        f"ROC-AUC={overall_roc_auc_text}"
    )


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

    train_age_groups = assign_age_groups(X_train_raw[AGE_COL])
    test_age_groups = assign_age_groups(X_test_raw[AGE_COL])

    overall_before_counts = class_counts(y_train_raw)
    overall_after_counts = {0: 0, 1: 0}
    cohort_results: Dict[str, Any] = {}

    final_pred = pd.Series(index=y_test_raw.index, dtype=float)
    final_prob = pd.Series(index=y_test_raw.index, dtype=float)

    for cohort in AGE_GROUPS:
        train_mask = train_age_groups == cohort
        test_mask = test_age_groups == cohort

        X_train_cohort_raw = X_train_raw.loc[train_mask]
        y_train_cohort_raw = y_train_raw.loc[train_mask].astype(int)
        X_test_cohort_raw = X_test_raw.loc[test_mask]
        y_test_cohort_raw = y_test_raw.loc[test_mask].astype(int)

        if len(X_test_cohort_raw) == 0:
            cohort_results[cohort] = {
                "rule": COHORT_RULES[cohort],
                "train_rows_before_sampling": int(len(y_train_cohort_raw)),
                "train_rows_after_sampling": int(len(y_train_cohort_raw)),
                "test_rows": 0,
                "feature_selection": feature_selection_not_applied(
                    reason="empty_test_cohort",
                    feature_names=[],
                ),
                "sampling": {
                    "applied": False,
                    "reason": "empty_test_cohort",
                    "before_counts": class_counts(y_train_cohort_raw),
                    "after_counts": class_counts(y_train_cohort_raw),
                    "generated_samples": 0,
                    "k_neighbors_used": 0,
                },
                "tuning": {
                    "method": "GridSearchCV",
                    "status": "skipped",
                    "reason": "empty_test_cohort",
                    "cv_folds": 0,
                    "param_grid_size": len(list(ParameterGrid(GRIDSEARCH_PARAM_GRID))),
                    "best_params": None,
                    "best_cv_score_f1": None,
                },
                "metrics": None,
            }
            continue

        if len(X_train_cohort_raw) == 0 or y_train_cohort_raw.nunique() < 2:
            base_prob = float(y_train_raw.mean())
            base_pred = int(base_prob >= 0.5)
            cohort_pred = np.full(len(X_test_cohort_raw), base_pred, dtype=int)
            cohort_prob = np.full(len(X_test_cohort_raw), base_prob, dtype=float)
            sampling_info = {
                "applied": False,
                "reason": "insufficient_train_cohort",
                "before_counts": class_counts(y_train_cohort_raw),
                "after_counts": class_counts(y_train_cohort_raw),
                "generated_samples": 0,
                "k_neighbors_used": 0,
            }
            tuning_info = {
                "method": "GridSearchCV",
                "status": "skipped",
                "reason": "insufficient_train_cohort",
                "cv_folds": 0,
                "param_grid_size": len(list(ParameterGrid(GRIDSEARCH_PARAM_GRID))),
                "best_params": None,
                "best_cv_score_f1": None,
            }
            metrics = compute_metrics(y_test_cohort_raw, cohort_pred, cohort_prob)
            final_pred.loc[X_test_cohort_raw.index] = cohort_pred
            final_prob.loc[X_test_cohort_raw.index] = cohort_prob

            for label, count in class_counts(y_train_cohort_raw).items():
                overall_after_counts[label] = overall_after_counts.get(label, 0) + count

            cohort_results[cohort] = {
                "rule": COHORT_RULES[cohort],
                "train_rows_before_sampling": int(len(y_train_cohort_raw)),
                "train_rows_after_sampling": int(len(y_train_cohort_raw)),
                "test_rows": int(len(y_test_cohort_raw)),
                "feature_selection": feature_selection_not_applied(
                    reason="insufficient_train_cohort",
                    feature_names=[],
                ),
                "sampling": sampling_info,
                "tuning": tuning_info,
                "metrics": metrics,
            }
            continue

        preprocessor = build_preprocessor()
        train_df = transform_with_preprocessor(
            X_train_cohort_raw,
            y_train_cohort_raw,
            preprocessor,
            fit=True,
        )
        test_df = transform_with_preprocessor(
            X_test_cohort_raw,
            y_test_cohort_raw,
            preprocessor,
            fit=False,
        )

        feature_cols = [col for col in train_df.columns if col != TARGET_COL]
        X_train_cohort = train_df[feature_cols]
        y_train_cohort = train_df[TARGET_COL].astype(int)
        X_test_cohort = test_df[feature_cols]
        y_test_cohort = test_df[TARGET_COL].astype(int)
        X_train_selected, X_test_selected, feature_selection_info = select_top_features(
            X_train_cohort,
            y_train_cohort,
            X_test_cohort,
        )

        if use_smote:
            X_train_final, y_train_final, sampling_info = smote_resample_binary(
                X_train_selected,
                y_train_cohort,
                random_state=RANDOM_STATE,
            )
        else:
            X_train_final = X_train_selected.reset_index(drop=True)
            y_train_final = y_train_cohort.reset_index(drop=True)
            sampling_info = {
                "applied": False,
                "reason": "smote_disabled",
                "before_counts": class_counts(y_train_cohort),
                "after_counts": class_counts(y_train_cohort),
                "generated_samples": 0,
                "k_neighbors_used": 0,
            }

        model, tuning_info = tune_xgboost_with_gridsearch(X_train_final, y_train_final)
        cohort_pred = model.predict(X_test_selected).astype(int)
        cohort_prob = model.predict_proba(X_test_selected)[:, 1]
        metrics = compute_metrics(y_test_cohort, cohort_pred, cohort_prob)

        final_pred.loc[X_test_cohort_raw.index] = cohort_pred
        final_prob.loc[X_test_cohort_raw.index] = cohort_prob

        for label, count in class_counts(y_train_final).items():
            overall_after_counts[label] = overall_after_counts.get(label, 0) + count

        cohort_results[cohort] = {
            "rule": COHORT_RULES[cohort],
            "train_rows_before_sampling": int(len(y_train_cohort)),
            "train_rows_after_sampling": int(len(y_train_final)),
            "test_rows": int(len(y_test_cohort_raw)),
            "feature_selection": feature_selection_info,
            "sampling": sampling_info,
            "tuning": tuning_info,
            "metrics": metrics,
        }

    missing_idx = final_pred[final_pred.isna()].index
    missing_prediction_fallback = None
    if len(missing_idx) > 0:
        fallback_prob = float(y_train_raw.mean())
        fallback_pred = int(fallback_prob >= 0.5)
        final_pred.loc[missing_idx] = fallback_pred
        final_prob.loc[missing_idx] = fallback_prob
        missing_prediction_fallback = {
            "rows_filled": int(len(missing_idx)),
            "fallback_prob": fallback_prob,
            "fallback_pred": fallback_pred,
        }

    ordered_idx = y_test_raw.index
    overall_metrics = compute_metrics(
        y_test_raw.loc[ordered_idx],
        final_pred.loc[ordered_idx].astype(int),
        final_prob.loc[ordered_idx].astype(float),
    )

    results: Dict[str, Any] = {
        "variant": variant_name,
        "use_smote": bool(use_smote),
        "dataset_rows_after_cleaning": int(len(df_clean)),
        "dataset_source": {
            "mode": "raw_dataset",
            "raw_file": str(raw_file),
        },
        "feature_selection": {
            "enabled": True,
            "method": "xgboost_feature_importance_top_k",
            "top_k": int(FEATURE_SELECTION_TOP_K),
            "stage": "before_smote_on_training_data",
        },
        "target_definition": "1 = cardiovascular disease, 0 = no cardiovascular disease",
        "cohort_definition": COHORT_RULES,
        "grid_search": {
            "enabled": True,
            "method": "GridSearchCV",
            "scoring": GRIDSEARCH_SCORING,
            "param_grid": GRIDSEARCH_PARAM_GRID,
            "param_grid_size": len(list(ParameterGrid(GRIDSEARCH_PARAM_GRID))),
        },
        "train_rows_before_sampling": int(len(y_train_raw)),
        "train_rows_after_sampling": int(sum(overall_after_counts.values())),
        "test_rows": int(len(y_test_raw)),
        "train_class_counts_before_sampling_overall": overall_before_counts,
        "train_class_counts_after_sampling_overall": overall_after_counts,
        "cohorts": cohort_results,
        "overall_metrics": overall_metrics,
        "missing_prediction_fallback": missing_prediction_fallback,
    }

    save_results(output_json, results)
    print_variant_summary(results)

    return results


def run_variant_on_generated_data(
    processed_train_file: Path,
    processed_test_file: Path,
    raw_reference_file: Path,
    use_smote: bool,
    output_json: Path | None,
    variant_name: str,
) -> Dict[str, Any]:
    train_df = load_processed_data(processed_train_file)
    test_df = load_processed_data(processed_test_file)
    age_cutoffs = compute_processed_age_cutoffs(raw_reference_file)
    young_upper_bound = age_cutoffs["young_upper_bound_processed"]
    middle_upper_bound = age_cutoffs["middle_upper_bound_processed"]

    feature_cols = [col for col in train_df.columns if col != TARGET_COL]
    y_train_full = train_df[TARGET_COL].astype(int)
    y_test_full = test_df[TARGET_COL].astype(int)

    train_age_groups = assign_age_groups(
        train_df[AGE_COL],
        young_upper_bound=young_upper_bound,
        middle_upper_bound=middle_upper_bound,
    )
    test_age_groups = assign_age_groups(
        test_df[AGE_COL],
        young_upper_bound=young_upper_bound,
        middle_upper_bound=middle_upper_bound,
    )

    overall_before_counts = class_counts(y_train_full)
    overall_after_counts = {0: 0, 1: 0}
    cohort_results: Dict[str, Any] = {}

    final_pred = pd.Series(index=test_df.index, dtype=float)
    final_prob = pd.Series(index=test_df.index, dtype=float)

    for cohort in AGE_GROUPS:
        train_mask = train_age_groups == cohort
        test_mask = test_age_groups == cohort

        X_train_cohort = train_df.loc[train_mask, feature_cols].reset_index(drop=True)
        y_train_cohort = train_df.loc[train_mask, TARGET_COL].astype(int).reset_index(drop=True)
        X_test_cohort = test_df.loc[test_mask, feature_cols]
        y_test_cohort = test_df.loc[test_mask, TARGET_COL].astype(int)

        if len(X_test_cohort) == 0:
            cohort_results[cohort] = {
                "rule": COHORT_RULES[cohort],
                "train_rows_before_sampling": int(len(y_train_cohort)),
                "train_rows_after_sampling": int(len(y_train_cohort)),
                "test_rows": 0,
                "feature_selection": feature_selection_not_applied(
                    reason="empty_test_cohort",
                    feature_names=[],
                ),
                "sampling": {
                    "applied": False,
                    "reason": "empty_test_cohort",
                    "before_counts": class_counts(y_train_cohort),
                    "after_counts": class_counts(y_train_cohort),
                    "generated_samples": 0,
                    "k_neighbors_used": 0,
                },
                "tuning": {
                    "method": "GridSearchCV",
                    "status": "skipped",
                    "reason": "empty_test_cohort",
                    "cv_folds": 0,
                    "param_grid_size": len(list(ParameterGrid(GRIDSEARCH_PARAM_GRID))),
                    "best_params": None,
                    "best_cv_score_f1": None,
                },
                "metrics": None,
            }
            continue

        if len(X_train_cohort) == 0 or y_train_cohort.nunique() < 2:
            base_prob = float(y_train_full.mean())
            base_pred = int(base_prob >= 0.5)
            cohort_pred = np.full(len(X_test_cohort), base_pred, dtype=int)
            cohort_prob = np.full(len(X_test_cohort), base_prob, dtype=float)
            sampling_info = {
                "applied": False,
                "reason": "insufficient_train_cohort",
                "before_counts": class_counts(y_train_cohort),
                "after_counts": class_counts(y_train_cohort),
                "generated_samples": 0,
                "k_neighbors_used": 0,
            }
            tuning_info = {
                "method": "GridSearchCV",
                "status": "skipped",
                "reason": "insufficient_train_cohort",
                "cv_folds": 0,
                "param_grid_size": len(list(ParameterGrid(GRIDSEARCH_PARAM_GRID))),
                "best_params": None,
                "best_cv_score_f1": None,
            }
            metrics = compute_metrics(y_test_cohort, cohort_pred, cohort_prob)
            final_pred.loc[X_test_cohort.index] = cohort_pred
            final_prob.loc[X_test_cohort.index] = cohort_prob

            for label, count in class_counts(y_train_cohort).items():
                overall_after_counts[label] = overall_after_counts.get(label, 0) + count

            cohort_results[cohort] = {
                "rule": COHORT_RULES[cohort],
                "train_rows_before_sampling": int(len(y_train_cohort)),
                "train_rows_after_sampling": int(len(y_train_cohort)),
                "test_rows": int(len(y_test_cohort)),
                "feature_selection": feature_selection_not_applied(
                    reason="insufficient_train_cohort",
                    feature_names=[],
                ),
                "sampling": sampling_info,
                "tuning": tuning_info,
                "metrics": metrics,
            }
            continue

        X_train_selected, X_test_selected, feature_selection_info = select_top_features(
            X_train_cohort,
            y_train_cohort,
            X_test_cohort,
        )

        if use_smote:
            X_train_final, y_train_final, sampling_info = smote_resample_binary(
                X_train_selected,
                y_train_cohort,
                random_state=RANDOM_STATE,
            )
        else:
            X_train_final = X_train_selected.reset_index(drop=True)
            y_train_final = y_train_cohort.reset_index(drop=True)
            sampling_info = {
                "applied": False,
                "reason": "smote_disabled",
                "before_counts": class_counts(y_train_cohort),
                "after_counts": class_counts(y_train_cohort),
                "generated_samples": 0,
                "k_neighbors_used": 0,
            }

        model, tuning_info = tune_xgboost_with_gridsearch(X_train_final, y_train_final)
        cohort_pred = model.predict(X_test_selected).astype(int)
        cohort_prob = model.predict_proba(X_test_selected)[:, 1]
        metrics = compute_metrics(y_test_cohort, cohort_pred, cohort_prob)

        final_pred.loc[X_test_cohort.index] = cohort_pred
        final_prob.loc[X_test_cohort.index] = cohort_prob

        for label, count in class_counts(y_train_final).items():
            overall_after_counts[label] = overall_after_counts.get(label, 0) + count

        cohort_results[cohort] = {
            "rule": COHORT_RULES[cohort],
            "train_rows_before_sampling": int(len(y_train_cohort)),
            "train_rows_after_sampling": int(len(y_train_final)),
            "test_rows": int(len(y_test_cohort)),
            "feature_selection": feature_selection_info,
            "sampling": sampling_info,
            "tuning": tuning_info,
            "metrics": metrics,
        }

    missing_idx = final_pred[final_pred.isna()].index
    missing_prediction_fallback = None
    if len(missing_idx) > 0:
        fallback_prob = float(y_train_full.mean())
        fallback_pred = int(fallback_prob >= 0.5)
        final_pred.loc[missing_idx] = fallback_pred
        final_prob.loc[missing_idx] = fallback_prob
        missing_prediction_fallback = {
            "rows_filled": int(len(missing_idx)),
            "fallback_prob": fallback_prob,
            "fallback_pred": fallback_pred,
        }

    ordered_idx = test_df.index
    overall_metrics = compute_metrics(
        y_test_full.loc[ordered_idx],
        final_pred.loc[ordered_idx].astype(int),
        final_prob.loc[ordered_idx].astype(float),
    )

    results: Dict[str, Any] = {
        "variant": variant_name,
        "use_smote": bool(use_smote),
        "dataset_rows_loaded": {
            "train": int(len(train_df)),
            "test": int(len(test_df)),
        },
        "dataset_source": {
            "mode": "generated_processed_train_with_processed_test",
            "processed_train_file": str(processed_train_file),
            "processed_test_file": str(processed_test_file),
            "raw_reference_file": str(raw_reference_file),
            "age_cutoffs_processed": age_cutoffs,
        },
        "feature_selection": {
            "enabled": True,
            "method": "xgboost_feature_importance_top_k",
            "top_k": int(FEATURE_SELECTION_TOP_K),
            "stage": "before_smote_on_training_data",
        },
        "target_definition": "1 = cardiovascular disease, 0 = no cardiovascular disease",
        "cohort_definition": COHORT_RULES,
        "grid_search": {
            "enabled": True,
            "method": "GridSearchCV",
            "scoring": GRIDSEARCH_SCORING,
            "param_grid": GRIDSEARCH_PARAM_GRID,
            "param_grid_size": len(list(ParameterGrid(GRIDSEARCH_PARAM_GRID))),
        },
        "train_rows_before_sampling": int(len(y_train_full)),
        "train_rows_after_sampling": int(sum(overall_after_counts.values())),
        "test_rows": int(len(y_test_full)),
        "train_class_counts_before_sampling_overall": overall_before_counts,
        "train_class_counts_after_sampling_overall": overall_after_counts,
        "cohorts": cohort_results,
        "overall_metrics": overall_metrics,
        "missing_prediction_fallback": missing_prediction_fallback,
    }

    save_results(output_json, results)
    print_variant_summary(results)

    return results
