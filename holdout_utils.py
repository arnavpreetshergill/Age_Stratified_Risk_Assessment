from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, StratifiedKFold

from training_utils import (
    FEATURE_SELECTION_TOP_K,
    TUNING_SCORING,
    build_model,
    class_counts,
    default_model_params,
    get_model_display_name,
    get_model_param_keys,
    get_tuning_grid,
    get_tuning_iterations,
    select_top_features,
    smote_resample_binary,
)


def default_model_info(
    label: str,
    reason: str = "fixed_default_model",
    status: str = "applied",
    model_key: str = "xgboost",
    params: Dict[str, Any] | None = None,
    method: str | None = None,
    tuned: bool = False,
    best_score: float | None = None,
    cv_folds: int | None = None,
    search_candidates: int = 0,
    search_iterations: int = 0,
) -> Dict[str, Any]:
    resolved_method = method or ("randomized_search_cv" if tuned else "default_model_params")
    return {
        "model_key": model_key,
        "model_name": get_model_display_name(model_key),
        "method": resolved_method,
        "status": status,
        "reason": reason,
        "label": label,
        "tuned": bool(tuned),
        "params": default_model_params(model_key) if params is None else params,
        "best_score": None if best_score is None or pd.isna(best_score) else float(best_score),
        "cv_folds": None if cv_folds is None else int(cv_folds),
        "search_candidates": int(search_candidates),
        "search_iterations": int(search_iterations),
        "scoring": TUNING_SCORING,
    }


def _fit_with_warning_suppression(model, X_train: pd.DataFrame, y_train: pd.Series):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        model.fit(X_train, y_train.astype(int))
    return model


def _predict_probabilities(model, X_eval: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_eval)[:, 1]
        return np.asarray(prob, dtype=float)

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X_eval), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision))

    pred = np.asarray(model.predict(X_eval), dtype=float)
    return pred


def _usable_cv_folds(y_train: pd.Series) -> int:
    counts = y_train.astype(int).value_counts()
    if counts.empty:
        return 0
    min_class_count = int(counts.min())
    if min_class_count < 2:
        return 0
    return min(3, min_class_count)


def _selected_model_params(model_key: str, estimator) -> Dict[str, Any]:
    estimator_params = estimator.get_params(deep=False)
    selected = {}
    for param_key in get_model_param_keys(model_key):
        if param_key not in estimator_params:
            continue
        value = estimator_params[param_key]
        if isinstance(value, np.generic):
            value = value.item()
        selected[param_key] = value
    return selected


def _tune_model(
    model_key: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    label: str,
    random_state: int,
    tune_hyperparameters: bool,
) -> tuple[Any, Dict[str, Any]]:
    default_estimator = build_model(model_key=model_key, random_state=random_state)
    default_info = default_model_info(
        label=label,
        model_key=model_key,
        params=_selected_model_params(model_key, default_estimator),
    )

    if not tune_hyperparameters:
        default_info["reason"] = "hyperparameter_tuning_disabled"
        return default_estimator, default_info

    if y_train.nunique() < 2:
        default_info["reason"] = "single_class_train_data"
        default_info["status"] = "fallback_default"
        return default_estimator, default_info

    cv_folds = _usable_cv_folds(y_train)
    if cv_folds < 2:
        default_info["reason"] = "insufficient_minority_samples_for_cv"
        default_info["status"] = "fallback_default"
        return default_estimator, default_info

    tuning_grid = get_tuning_grid(model_key)
    candidate_count = len(list(ParameterGrid(tuning_grid)))
    search_iterations = min(get_tuning_iterations(model_key), candidate_count)
    if search_iterations <= 0:
        default_info["reason"] = "empty_search_space"
        default_info["status"] = "fallback_default"
        return default_estimator, default_info

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=build_model(model_key=model_key, random_state=random_state),
        param_distributions=tuning_grid,
        n_iter=search_iterations,
        scoring=TUNING_SCORING,
        cv=cv,
        refit=False,
        random_state=random_state,
        n_jobs=1,
        error_score=np.nan,
    )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            search.fit(X_train, y_train.astype(int))
        best_params = dict(search.best_params_)
        tuned_estimator = build_model(model_key=model_key, random_state=random_state, params=best_params)
        tuned_info = default_model_info(
            label=label,
            reason="hyperparameter_search_completed",
            status="tuned",
            model_key=model_key,
            params=_selected_model_params(model_key, tuned_estimator),
            tuned=True,
            best_score=float(search.best_score_),
            cv_folds=cv_folds,
            search_candidates=candidate_count,
            search_iterations=search_iterations,
        )
        return tuned_estimator, tuned_info
    except Exception as exc:
        default_info["reason"] = f"tuning_failed:{type(exc).__name__}"
        default_info["status"] = "fallback_default"
        default_info["cv_folds"] = cv_folds
        default_info["search_candidates"] = candidate_count
        default_info["search_iterations"] = search_iterations
        return default_estimator, default_info


def fit_training_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    use_smote: bool,
    top_k: int = FEATURE_SELECTION_TOP_K,
    random_state: int = 42,
    model_key: str = "xgboost",
    label: str | None = None,
    tune_hyperparameters: bool = False,
) -> Dict[str, Any]:
    X_train_selected, X_eval_selected, feature_info = select_top_features(
        X_train,
        y_train,
        X_eval,
        top_k=top_k,
        random_state=random_state,
    )

    if use_smote:
        X_train_final, y_train_final, sampling_info = smote_resample_binary(
            X_train_selected,
            y_train,
            random_state=random_state,
        )
    else:
        X_train_final = X_train_selected.reset_index(drop=True)
        y_train_final = y_train.astype(int).reset_index(drop=True)
        before_counts = class_counts(y_train_final)
        sampling_info = {
            "applied": False,
            "reason": "smote_disabled",
            "before_counts": before_counts,
            "after_counts": before_counts.copy(),
            "generated_samples": 0,
            "k_neighbors_used": 0,
        }

    resolved_label = label or f"{get_model_display_name(model_key)} Holdout"
    model, model_info = _tune_model(
        model_key=model_key,
        X_train=X_train_final,
        y_train=y_train_final,
        label=resolved_label,
        random_state=random_state,
        tune_hyperparameters=tune_hyperparameters,
    )

    model = _fit_with_warning_suppression(model, X_train_final, y_train_final)
    pred = model.predict(X_eval_selected).astype(int)
    prob = _predict_probabilities(model, X_eval_selected)

    return {
        "model": model,
        "model_info": model_info,
        "pred": np.asarray(pred, dtype=int),
        "prob": np.asarray(prob, dtype=float),
        "feature_selection": feature_info,
        "sampling": sampling_info,
        "X_eval_selected": X_eval_selected,
        "X_train_selected": X_train_selected,
        "X_train_final": X_train_final,
        "y_train_final": y_train_final,
    }


def fit_model_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    label: str,
    model_key: str,
    top_k: int = FEATURE_SELECTION_TOP_K,
    use_smote: bool = True,
    random_state: int = 42,
    tune_hyperparameters: bool = False,
) -> Dict[str, Any]:
    return fit_training_pipeline(
        X_train,
        y_train,
        X_eval,
        use_smote=use_smote,
        top_k=top_k,
        random_state=random_state,
        model_key=model_key,
        label=label,
        tune_hyperparameters=tune_hyperparameters,
    )


def fit_xgboost_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    label: str,
    top_k: int = FEATURE_SELECTION_TOP_K,
    use_smote: bool = True,
    random_state: int = 42,
    tune_hyperparameters: bool = False,
) -> Dict[str, Any]:
    return fit_model_pipeline(
        X_train,
        y_train,
        X_eval,
        label=label,
        model_key="xgboost",
        top_k=top_k,
        use_smote=use_smote,
        random_state=random_state,
        tune_hyperparameters=tune_hyperparameters,
    )


def fit_final_xgboost_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    label: str,
    top_k: int = FEATURE_SELECTION_TOP_K,
    use_smote: bool = True,
    random_state: int = 42,
    X_reference: pd.DataFrame | None = None,
    tune_hyperparameters: bool = False,
) -> Dict[str, Any]:
    reference = X_train if X_reference is None else X_reference
    return fit_xgboost_pipeline(
        X_train,
        y_train,
        reference,
        label=label,
        top_k=top_k,
        use_smote=use_smote,
        random_state=random_state,
        tune_hyperparameters=tune_hyperparameters,
    )
