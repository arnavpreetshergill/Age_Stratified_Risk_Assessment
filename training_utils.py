from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.neighbors import NearestNeighbors

FEATURE_SELECTION_TOP_K = 10
BINARY_FEATURES = {"sex", "fasting_blood_sugar", "exercise_angina"}
ONE_HOT_GROUPS = [
    ["chest_pain_type_2", "chest_pain_type_3", "chest_pain_type_4"],
    ["resting_ecg_1", "resting_ecg_2"],
    ["st_slope_2", "st_slope_3"],
]


def build_default_model(random_state: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.1,
        subsample=1.0,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=1,
    )


def tune_xgboost_with_gridsearch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    label: str,
    param_grid: Dict[str, list[Any]],
    scoring: str = "f1",
    max_cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    y_work = y_train.astype(int).reset_index(drop=True)
    counts = y_work.value_counts()
    min_class_count = int(counts.min()) if counts.size > 0 else 0
    param_grid_size = len(list(ParameterGrid(param_grid)))

    if counts.size < 2 or min_class_count < 2:
        model = build_default_model(random_state=random_state)
        model.fit(X_train, y_work)
        fallback_params = {
            "n_estimators": int(model.get_params()["n_estimators"]),
            "max_depth": int(model.get_params()["max_depth"]),
            "learning_rate": float(model.get_params()["learning_rate"]),
            "subsample": float(model.get_params()["subsample"]),
        }
        print(
            f"   -> {label}: fallback to default params {fallback_params} "
            f"(insufficient class diversity for CV)"
        )
        return model, {
            "status": "fallback_default_model",
            "reason": "insufficient_class_diversity_for_cv",
            "cv_folds": 0,
            "param_grid_size": param_grid_size,
            "best_params": fallback_params,
            "best_cv_score_f1": None,
        }

    cv_folds = min(max_cv_folds, min_class_count)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    search = GridSearchCV(
        estimator=build_default_model(random_state=random_state),
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_work)

    best_params = {
        "n_estimators": int(search.best_params_["n_estimators"]),
        "max_depth": int(search.best_params_["max_depth"]),
        "learning_rate": float(search.best_params_["learning_rate"]),
        "subsample": float(search.best_params_["subsample"]),
    }

    print(
        f"   -> {label}: best params {best_params} "
        f"(CV f1={search.best_score_:.4f}, folds={cv_folds}, grid={param_grid_size})"
    )

    return search.best_estimator_, {
        "status": "tuned",
        "reason": None,
        "cv_folds": int(cv_folds),
        "param_grid_size": param_grid_size,
        "best_params": best_params,
        "best_cv_score_f1": float(search.best_score_),
    }


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


def smote_resample_binary(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
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


def select_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    top_k: int = FEATURE_SELECTION_TOP_K,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    candidate_features = list(X_train.columns)
    if len(candidate_features) <= top_k:
        info = {
            "applied": False,
            "reason": "feature_count_within_limit",
            "method": "xgboost_feature_importance_top_k",
            "requested_top_k": int(top_k),
            "selected_features": candidate_features,
            "selected_feature_importances": None,
        }
        return X_train.copy(), X_test.loc[:, candidate_features].copy(), info

    if y_train.nunique() < 2:
        info = {
            "applied": False,
            "reason": "single_class_train_data",
            "method": "xgboost_feature_importance_top_k",
            "requested_top_k": int(top_k),
            "selected_features": candidate_features,
            "selected_feature_importances": None,
        }
        return X_train.copy(), X_test.loc[:, candidate_features].copy(), info

    selector_model = build_default_model(random_state=random_state)
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
        "selected_features": selected_features,
        "selected_feature_importances": feature_importances,
    }
    return (
        X_train.loc[:, selected_features].copy(),
        X_test.loc[:, selected_features].copy(),
        info,
    )
