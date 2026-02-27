import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, train_test_split

from dataGeneration import augment_processed_data
from preprocess import (
    NUMERIC_FEATURES,
    TARGET_COL,
    build_preprocessor,
    load_and_clean_data,
    transform_with_preprocessor,
)

# --- CONFIGURATION ---
RAW_FILE = "heart_statlog_cleveland_hungary_final(1).csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_TRAIN_SIZE = 100000
AGE_GROUPS = ["Young", "Middle", "Elderly"]
GRIDSEARCH_SCORING = "f1"
MAX_GRIDSEARCH_FOLDS = 5
GRIDSEARCH_PARAM_GRID = {
    "n_estimators": [80, 140],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.15],
    "subsample": [0.8, 1.0],
}


def build_default_model():
    return xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.1,
        subsample=1.0,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=1,
    )


def tune_xgboost_with_gridsearch(X_train, y_train, label):
    y_work = y_train.astype(int).reset_index(drop=True)
    counts = y_work.value_counts()
    min_class_count = int(counts.min()) if counts.size > 0 else 0
    param_grid_size = len(list(ParameterGrid(GRIDSEARCH_PARAM_GRID)))

    if counts.size < 2 or min_class_count < 2:
        model = build_default_model()
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

    cv_folds = min(MAX_GRIDSEARCH_FOLDS, min_class_count)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=build_default_model(),
        param_grid=GRIDSEARCH_PARAM_GRID,
        scoring=GRIDSEARCH_SCORING,
        cv=cv,
        n_jobs=-1,
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


def get_age_cutoffs_from_preprocessor(preprocessor, low_age=45, high_age=65):
    scaler = preprocessor.named_transformers_["num"]
    age_idx = NUMERIC_FEATURES.index("age")
    age_mean = scaler.mean_[age_idx]
    age_std = scaler.scale_[age_idx]
    z45 = (low_age - age_mean) / age_std
    z65 = (high_age - age_mean) / age_std
    return z45, z65, age_mean, age_std


def assign_age_group(age_series, z45, z65):
    conditions = [
        age_series < z45,
        (age_series >= z45) & (age_series <= z65),
        age_series > z65,
    ]
    choices = ["Young", "Middle", "Elderly"]
    return np.select(conditions, choices, default="Unknown")


def compute_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if np.unique(y_true).size > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    return metrics


def fmt(x):
    return "N/A" if pd.isna(x) else f"{x:.4f}"


def fmt_cv(x):
    if x is None:
        return "N/A"
    return "N/A" if pd.isna(x) else f"{x:.4f}"


def compare_strategies():
    print("=" * 70)
    print("LEAKAGE-SAFE MODEL COMPARISON: BASELINE vs AGE-STRATIFIED")
    print("=" * 70)

    if not os.path.exists(RAW_FILE):
        print(f"Error: '{RAW_FILE}' not found.")
        return

    # 1) Clean raw data and split first to avoid leakage.
    raw_df = load_and_clean_data(RAW_FILE)
    X_raw = raw_df.drop(columns=[TARGET_COL])
    y_raw = raw_df[TARGET_COL].astype(int)

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw,
        y_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_raw,
    )

    # 2) Fit preprocessing on train only.
    preprocessor = build_preprocessor()
    train_df = transform_with_preprocessor(X_train_raw, y_train_raw, preprocessor, fit=True)
    test_df = transform_with_preprocessor(X_test_raw, y_test_raw, preprocessor, fit=False)

    z45, z65, age_mean, age_std = get_age_cutoffs_from_preprocessor(preprocessor)
    train_df["age_group"] = assign_age_group(train_df["age"], z45, z65)
    test_df["age_group"] = assign_age_group(test_df["age"], z45, z65)

    print(f"Raw cleaned rows: {len(raw_df)}")
    print(f"Train/Test rows: {len(train_df)} / {len(test_df)}")
    print(
        f"Age thresholds in z-space from train scaler: "
        f"45y={z45:.3f}, 65y={z65:.3f} (mean={age_mean:.3f}, std={age_std:.3f})"
    )

    # 3) Augment only training data.
    train_aug = augment_processed_data(
        train_df.drop(columns=["age_group"]),
        target_total=TARGET_TRAIN_SIZE,
        age_z_45=z45,
        age_z_65=z65,
        random_state=RANDOM_STATE,
    )
    train_aug["age_group"] = assign_age_group(train_aug["age"], z45, z65)
    print(f"Augmented train rows: {len(train_aug)}")
    print(f"Train age groups: {train_aug['age_group'].value_counts().to_dict()}")
    print(f"Test age groups:  {test_df['age_group'].value_counts().to_dict()}")
    print("-" * 70)

    feature_cols = [c for c in train_aug.columns if c not in [TARGET_COL, "age_group"]]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(int)

    # 4) Baseline model.
    print("1) TUNING + TRAINING BASELINE model...")
    baseline_model, baseline_tuning = tune_xgboost_with_gridsearch(
        train_aug[feature_cols],
        train_aug[TARGET_COL],
        "Baseline",
    )
    baseline_pred = pd.Series(
        baseline_model.predict(X_test).astype(int),
        index=test_df.index,
    )
    baseline_prob = pd.Series(
        baseline_model.predict_proba(X_test)[:, 1],
        index=test_df.index,
    )
    baseline_overall = compute_metrics(y_test, baseline_pred, baseline_prob)

    baseline_group_metrics = {}
    for grp in AGE_GROUPS:
        mask = test_df["age_group"] == grp
        baseline_group_metrics[grp] = compute_metrics(
            y_test[mask],
            baseline_pred[mask],
            baseline_prob[mask],
        )

    # 5) Stratified specialist models.
    print("2) TUNING + TRAINING STRATIFIED specialists...")
    strat_pred = pd.Series(index=test_df.index, dtype=float)
    strat_prob = pd.Series(index=test_df.index, dtype=float)
    strat_group_metrics = {}
    strat_tuning = {}

    for grp in AGE_GROUPS:
        train_grp = train_aug[train_aug["age_group"] == grp]
        test_grp = test_df[test_df["age_group"] == grp]

        if train_grp.empty or test_grp.empty or train_grp[TARGET_COL].nunique() < 2:
            print(f"   -> {grp}: skipped (insufficient train/test class coverage)")
            strat_group_metrics[grp] = {
                "accuracy": np.nan,
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "tp": 0,
            }
            strat_tuning[grp] = {
                "status": "skipped",
                "reason": "insufficient_train_test_class_coverage",
                "cv_folds": 0,
                "param_grid_size": len(list(ParameterGrid(GRIDSEARCH_PARAM_GRID))),
                "best_params": None,
                "best_cv_score_f1": None,
            }
            continue

        specialist, tuning = tune_xgboost_with_gridsearch(
            train_grp[feature_cols],
            train_grp[TARGET_COL],
            grp,
        )
        grp_pred = specialist.predict(test_grp[feature_cols]).astype(int)
        grp_prob = specialist.predict_proba(test_grp[feature_cols])[:, 1]

        strat_pred.loc[test_grp.index] = grp_pred
        strat_prob.loc[test_grp.index] = grp_prob
        strat_group_metrics[grp] = compute_metrics(
            test_grp[TARGET_COL].astype(int),
            grp_pred,
            grp_prob,
        )
        strat_tuning[grp] = tuning
        print(f"   -> {grp}: Accuracy {strat_group_metrics[grp]['accuracy']:.4f}")

    # Fallback if any rows were not scored by specialists.
    missing_idx = strat_pred[strat_pred.isna()].index
    if len(missing_idx) > 0:
        print(f"   -> Warning: {len(missing_idx)} test rows missing specialist predictions, using baseline.")
        strat_pred.loc[missing_idx] = baseline_pred.loc[missing_idx]
        strat_prob.loc[missing_idx] = baseline_prob.loc[missing_idx]

    strat_pred = strat_pred.astype(int)
    strat_overall = compute_metrics(y_test, strat_pred, strat_prob)

    # 6) Reporting.
    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print(f"{'Metric':<12} | {'Baseline':<10} | {'Stratified':<10} | {'Winner':<10}")
    print("-" * 70)
    for metric in ["accuracy", "roc_auc", "pr_auc", "recall", "f1"]:
        b = baseline_overall[metric]
        s = strat_overall[metric]
        if pd.isna(b) or pd.isna(s):
            winner = "N/A"
        elif s > b:
            winner = "Stratified"
        elif b > s:
            winner = "Baseline"
        else:
            winner = "Tie"
        print(f"{metric:<12} | {fmt(b):<10} | {fmt(s):<10} | {winner:<10}")

    print("-" * 70)
    print(
        f"Baseline CM [TN FP FN TP]: "
        f"[{baseline_overall['tn']} {baseline_overall['fp']} {baseline_overall['fn']} {baseline_overall['tp']}]"
    )
    print(
        f"Stratified CM [TN FP FN TP]: "
        f"[{strat_overall['tn']} {strat_overall['fp']} {strat_overall['fn']} {strat_overall['tp']}]"
    )

    print("\nPER-GROUP ACCURACY")
    print(f"{'Age Group':<10} | {'Baseline':<10} | {'Stratified':<10}")
    print("-" * 70)
    for grp in AGE_GROUPS:
        print(
            f"{grp:<10} | "
            f"{fmt(baseline_group_metrics[grp]['accuracy']):<10} | "
            f"{fmt(strat_group_metrics[grp]['accuracy']):<10}"
        )

    print("\nTUNING SUMMARY")
    print("-" * 70)
    print(
        f"Baseline best params: {baseline_tuning['best_params']} "
        f"| best CV f1: {fmt_cv(baseline_tuning['best_cv_score_f1'])}"
    )
    for grp in AGE_GROUPS:
        tuning = strat_tuning.get(grp, {})
        print(
            f"{grp:<10} params: {tuning.get('best_params')} "
            f"| best CV f1: {fmt_cv(tuning.get('best_cv_score_f1'))}"
        )


if __name__ == "__main__":
    compare_strategies()
