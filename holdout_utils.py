from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from training_utils import (
    FEATURE_SELECTION_TOP_K,
    build_model,
    class_counts,
    default_model_params,
    get_model_display_name,
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
) -> Dict[str, Any]:
    resolved_method = method or "default_model_params"
    return {
        "model_key": model_key,
        "model_name": get_model_display_name(model_key),
        "method": resolved_method,
        "status": status,
        "reason": reason,
        "label": label,
        "params": default_model_params(model_key) if params is None else params,
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


def _selected_model_params(model_key: str, estimator) -> Dict[str, Any]:
    estimator_params = estimator.get_params(deep=False)
    selected = {}
    for param_key in sorted(default_model_params(model_key).keys()):
        if param_key not in estimator_params:
            continue
        value = estimator_params[param_key]
        if isinstance(value, np.generic):
            value = value.item()
        selected[param_key] = value
    return selected


def fit_training_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    use_smote: bool,
    top_k: int = FEATURE_SELECTION_TOP_K,
    random_state: int = 42,
    model_key: str = "xgboost",
    label: str | None = None,
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

    model = build_model(model_key=model_key, random_state=random_state)
    resolved_label = label or f"{get_model_display_name(model_key)} Holdout"
    model_info = default_model_info(
        label=resolved_label,
        model_key=model_key,
        params=_selected_model_params(model_key, model),
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
    )


def fit_xgboost_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    label: str,
    top_k: int = FEATURE_SELECTION_TOP_K,
    use_smote: bool = True,
    random_state: int = 42,
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
    )


def fit_final_xgboost_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    label: str,
    top_k: int = FEATURE_SELECTION_TOP_K,
    use_smote: bool = True,
    random_state: int = 42,
    X_reference: pd.DataFrame | None = None,
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
    )
