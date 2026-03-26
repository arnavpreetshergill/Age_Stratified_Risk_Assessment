from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

FEATURE_SELECTION_TOP_K = 10
TUNING_SCORING = "roc_auc"
BINARY_FEATURES = {"sex", "fasting_blood_sugar", "exercise_angina"}
ONE_HOT_GROUPS = [
    ["chest_pain_type_2", "chest_pain_type_3", "chest_pain_type_4"],
    ["resting_ecg_1", "resting_ecg_2"],
    ["st_slope_2", "st_slope_3"],
]

LOGISTIC_REGRESSION_DEFAULT_PARAMS: Dict[str, Any] = {
    "C": 1.0,
    "solver": "liblinear",
    "max_iter": 2000,
}
RANDOM_FOREST_DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 250,
    "max_depth": 6,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": 1,
}
XGBOOST_DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 120,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "eval_metric": "logloss",
    "n_jobs": 1,
}
LIGHTGBM_DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "verbosity": -1,
    "n_jobs": 1,
}
SVM_DEFAULT_PARAMS: Dict[str, Any] = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "probability": True,
}
MLP_DEFAULT_PARAMS: Dict[str, Any] = {
    "hidden_layer_sizes": (64, 32),
    "alpha": 1e-4,
    "learning_rate_init": 1e-3,
    "max_iter": 1200,
    "early_stopping": True,
    "n_iter_no_change": 30,
}
KNN_DEFAULT_PARAMS: Dict[str, Any] = {
    "n_neighbors": 7,
    "weights": "distance",
    "p": 2,
}
DECISION_TREE_DEFAULT_PARAMS: Dict[str, Any] = {
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
}
NAIVE_BAYES_DEFAULT_PARAMS: Dict[str, Any] = {
    "var_smoothing": 1e-9,
}
CATBOOST_DEFAULT_PARAMS: Dict[str, Any] = {
    "iterations": 200,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "loss_function": "Logloss",
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": 1,
}
ADABOOST_DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 150,
    "learning_rate": 0.5,
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    default_params: Mapping[str, Any]
    tuning_grid: Mapping[str, list[Any]]
    builder: Callable[[int, Mapping[str, Any] | None], Any]
    tuning_iterations: int = 8
    dependency_error: str | None = None

    @property
    def available(self) -> bool:
        return self.dependency_error is None


def _merge_params(default_params: Mapping[str, Any], params: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    merged = dict(default_params)
    if params:
        merged.update(dict(params))
    return merged


def _build_logistic_regression(random_state: int, params: Mapping[str, Any] | None = None) -> LogisticRegression:
    config = _merge_params(LOGISTIC_REGRESSION_DEFAULT_PARAMS, params)
    config.setdefault("random_state", random_state)
    return LogisticRegression(**config)


def _build_random_forest(random_state: int, params: Mapping[str, Any] | None = None) -> RandomForestClassifier:
    config = _merge_params(RANDOM_FOREST_DEFAULT_PARAMS, params)
    config.setdefault("random_state", random_state)
    config.setdefault("n_jobs", 1)
    return RandomForestClassifier(**config)


def _build_xgboost(random_state: int, params: Mapping[str, Any] | None = None) -> xgb.XGBClassifier:
    config = _merge_params(XGBOOST_DEFAULT_PARAMS, params)
    config.setdefault("random_state", random_state)
    config.setdefault("eval_metric", "logloss")
    config.setdefault("n_jobs", 1)
    return xgb.XGBClassifier(**config)


def _build_lightgbm(random_state: int, params: Mapping[str, Any] | None = None):
    if LGBMClassifier is None:
        raise ImportError("lightgbm is not installed.")
    config = _merge_params(LIGHTGBM_DEFAULT_PARAMS, params)
    config.setdefault("random_state", random_state)
    config.setdefault("verbosity", -1)
    config.setdefault("n_jobs", 1)
    return LGBMClassifier(**config)


def _build_svm(random_state: int, params: Mapping[str, Any] | None = None) -> SVC:
    config = _merge_params(SVM_DEFAULT_PARAMS, params)
    config.setdefault("random_state", random_state)
    config.setdefault("probability", True)
    return SVC(**config)


def _build_mlp(random_state: int, params: Mapping[str, Any] | None = None) -> MLPClassifier:
    config = _merge_params(MLP_DEFAULT_PARAMS, params)
    config.setdefault("random_state", random_state)
    return MLPClassifier(**config)


def _build_knn(random_state: int, params: Mapping[str, Any] | None = None) -> KNeighborsClassifier:
    _ = random_state
    config = _merge_params(KNN_DEFAULT_PARAMS, params)
    return KNeighborsClassifier(**config)


def _build_decision_tree(random_state: int, params: Mapping[str, Any] | None = None) -> DecisionTreeClassifier:
    config = _merge_params(DECISION_TREE_DEFAULT_PARAMS, params)
    config.setdefault("random_state", random_state)
    return DecisionTreeClassifier(**config)


def _build_naive_bayes(random_state: int, params: Mapping[str, Any] | None = None) -> GaussianNB:
    _ = random_state
    config = _merge_params(NAIVE_BAYES_DEFAULT_PARAMS, params)
    return GaussianNB(**config)


def _build_catboost(random_state: int, params: Mapping[str, Any] | None = None):
    if CatBoostClassifier is None:
        raise ImportError("catboost is not installed.")
    config = _merge_params(CATBOOST_DEFAULT_PARAMS, params)
    config.setdefault("random_seed", random_state)
    config.setdefault("verbose", False)
    config.setdefault("allow_writing_files", False)
    config.setdefault("thread_count", 1)
    return CatBoostClassifier(**config)


def _build_adaboost(random_state: int, params: Mapping[str, Any] | None = None) -> AdaBoostClassifier:
    config = _merge_params(ADABOOST_DEFAULT_PARAMS, params)
    config.setdefault("random_state", random_state)
    return AdaBoostClassifier(**config)


def _build_model_specs() -> Dict[str, ModelSpec]:
    lightgbm_error = None if LGBMClassifier is not None else "lightgbm package is not installed."
    catboost_error = None if CatBoostClassifier is not None else "catboost package is not installed."

    return {
        "logistic_regression": ModelSpec(
            key="logistic_regression",
            display_name="Logistic Regression",
            default_params=LOGISTIC_REGRESSION_DEFAULT_PARAMS,
            tuning_grid={
                "C": [0.01, 0.1, 1.0, 5.0, 10.0],
                "solver": ["liblinear", "lbfgs"],
            },
            builder=_build_logistic_regression,
            tuning_iterations=8,
        ),
        "random_forest": ModelSpec(
            key="random_forest",
            display_name="Random Forest",
            default_params=RANDOM_FOREST_DEFAULT_PARAMS,
            tuning_grid={
                "n_estimators": [150, 250, 400],
                "max_depth": [4, 6, 8, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            builder=_build_random_forest,
            tuning_iterations=8,
        ),
        "xgboost": ModelSpec(
            key="xgboost",
            display_name="XGBoost",
            default_params=XGBOOST_DEFAULT_PARAMS,
            tuning_grid={
                "n_estimators": [80, 120, 180, 240],
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.03, 0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            builder=_build_xgboost,
            tuning_iterations=8,
        ),
        "lightgbm": ModelSpec(
            key="lightgbm",
            display_name="LightGBM",
            default_params=LIGHTGBM_DEFAULT_PARAMS,
            tuning_grid={
                "n_estimators": [120, 200, 300],
                "learning_rate": [0.03, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "max_depth": [-1, 4, 6, 8],
                "min_child_samples": [10, 20, 30],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            builder=_build_lightgbm,
            tuning_iterations=8,
            dependency_error=lightgbm_error,
        ),
        "svm": ModelSpec(
            key="svm",
            display_name="Support Vector Machine (SVM)",
            default_params=SVM_DEFAULT_PARAMS,
            tuning_grid={
                "C": [0.1, 1.0, 10.0, 25.0],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"],
            },
            builder=_build_svm,
            tuning_iterations=8,
        ),
        "mlp": ModelSpec(
            key="mlp",
            display_name="Artificial Neural Network (MLP)",
            default_params=MLP_DEFAULT_PARAMS,
            tuning_grid={
                "hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                "learning_rate_init": [1e-4, 5e-4, 1e-3, 1e-2],
            },
            builder=_build_mlp,
            tuning_iterations=8,
        ),
        "knn": ModelSpec(
            key="knn",
            display_name="K-Nearest Neighbors (KNN)",
            default_params=KNN_DEFAULT_PARAMS,
            tuning_grid={
                "n_neighbors": [3, 5, 7, 9, 11, 15],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
            builder=_build_knn,
            tuning_iterations=8,
        ),
        "decision_tree": ModelSpec(
            key="decision_tree",
            display_name="Decision Tree",
            default_params=DECISION_TREE_DEFAULT_PARAMS,
            tuning_grid={
                "criterion": ["gini", "entropy"],
                "max_depth": [3, 5, 7, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            builder=_build_decision_tree,
            tuning_iterations=8,
        ),
        "naive_bayes": ModelSpec(
            key="naive_bayes",
            display_name="Naive Bayes",
            default_params=NAIVE_BAYES_DEFAULT_PARAMS,
            tuning_grid={
                "var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
            },
            builder=_build_naive_bayes,
            tuning_iterations=5,
        ),
        "catboost": ModelSpec(
            key="catboost",
            display_name="CatBoost",
            default_params=CATBOOST_DEFAULT_PARAMS,
            tuning_grid={
                "iterations": [120, 200, 320],
                "depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.1],
                "l2_leaf_reg": [1.0, 3.0, 5.0, 7.0],
            },
            builder=_build_catboost,
            tuning_iterations=8,
            dependency_error=catboost_error,
        ),
        "adaboost": ModelSpec(
            key="adaboost",
            display_name="AdaBoost",
            default_params=ADABOOST_DEFAULT_PARAMS,
            tuning_grid={
                "n_estimators": [50, 100, 150, 250],
                "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
            },
            builder=_build_adaboost,
            tuning_iterations=8,
        ),
    }


MODEL_SPECS = _build_model_specs()
MODEL_ORDER = list(MODEL_SPECS.keys())


def get_model_spec(model_key: str) -> ModelSpec:
    if model_key not in MODEL_SPECS:
        raise KeyError(f"Unsupported model key: {model_key}")
    return MODEL_SPECS[model_key]


def get_model_specs(include_unavailable: bool = True) -> list[ModelSpec]:
    specs = [MODEL_SPECS[key] for key in MODEL_ORDER]
    if include_unavailable:
        return specs
    return [spec for spec in specs if spec.available]


def get_available_model_keys() -> list[str]:
    return [spec.key for spec in get_model_specs(include_unavailable=False)]


def get_unavailable_models() -> Dict[str, str]:
    return {
        spec.key: str(spec.dependency_error)
        for spec in get_model_specs(include_unavailable=True)
        if not spec.available
    }


def get_model_display_name(model_key: str) -> str:
    return get_model_spec(model_key).display_name


def get_tuning_grid(model_key: str) -> Dict[str, list[Any]]:
    return {key: list(values) for key, values in get_model_spec(model_key).tuning_grid.items()}


def get_tuning_iterations(model_key: str) -> int:
    return int(get_model_spec(model_key).tuning_iterations)


def get_model_param_keys(model_key: str) -> list[str]:
    spec = get_model_spec(model_key)
    param_keys = set(spec.default_params.keys()) | set(spec.tuning_grid.keys())
    return sorted(param_keys)


def default_model_params(model_key: str = "xgboost") -> Dict[str, Any]:
    return dict(get_model_spec(model_key).default_params)


def build_model(
    model_key: str = "xgboost",
    random_state: int = 42,
    params: Mapping[str, Any] | None = None,
):
    spec = get_model_spec(model_key)
    if not spec.available:
        raise ImportError(spec.dependency_error or f"{model_key} is unavailable.")
    return spec.builder(random_state, params)


def build_default_model(random_state: int = 42) -> xgb.XGBClassifier:
    return build_model("xgboost", random_state=random_state)


def class_counts(y: pd.Series) -> Dict[int, int]:
    counts = y.astype(int).value_counts().sort_index()
    count_map = {0: 0, 1: 0}
    for label, count in counts.items():
        count_map[int(label)] = int(count)
    return count_map


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
