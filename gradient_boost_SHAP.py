import os
import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, train_test_split

# --- CONFIGURATION ---
# List of split files and friendly names
DATASETS = [
    {"file": "heart_data_young.csv", "name": "Young Adults (<45)"},
    {"file": "heart_data_middle.csv", "name": "Middle-Aged (45-65)"},
    {"file": "heart_data_elderly.csv", "name": "Elderly (>65)"},
]
RANDOM_STATE = 42
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


def tune_xgboost_with_gridsearch(X_train, y_train, group_name):
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
            f"Tuning fallback for {group_name}: {fallback_params} "
            f"(insufficient class diversity for CV)"
        )
        return model

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
        f"Tuned params for {group_name}: {best_params} "
        f"(CV f1={search.best_score_:.4f}, folds={cv_folds}, grid={param_grid_size})"
    )
    return search.best_estimator_


def analyze_group(file_path, group_name):
    print("=" * 60)
    print(f"STARTING ANALYSIS: {group_name}")
    print("=" * 60)

    # 1. Load Data
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found. Skipping.")
        return

    df = pd.read_csv(file_path)
    print(f"Data Loaded: {len(df)} rows")
    if len(df) < 10 or df["target"].nunique() < 2:
        print(f"Skipping {group_name}: not enough rows/classes for train/test split.")
        return

    # 2. Prepare X and y
    target_col = "target"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 4. Tune + train Gradient Boosting Model (XGBoost)
    model = tune_xgboost_with_gridsearch(X_train, y_train, group_name)

    # 5. Evaluate Performance
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy for {group_name}: {acc:.2%}")
    print("-" * 30)

    # 6. SHAP Analysis
    print(f"Generating SHAP plots for {group_name}...")

    # Create object that can calculate SHAP values
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    # Plot 1: Summary plot
    plt.figure(figsize=(10, 6))
    plt.title(f"SHAP Summary: {group_name}")
    shap.summary_plot(shap_values, X_test, show=False)

    safe_name = group_name.split()[0]  # e.g., "Young"
    plot_filename = f"shap_summary_{safe_name}.png"
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_filename}")

    # Plot 2: Bar plot (feature importance)
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance: {group_name}")
    shap.plots.bar(shap_values, show=False)

    bar_filename = f"shap_importance_{safe_name}.png"
    plt.savefig(bar_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved: {bar_filename}")
    print("\n")


# --- EXECUTION LOOP ---
if __name__ == "__main__":
    for data in DATASETS:
        analyze_group(data["file"], data["name"])

    print("Done! Check your folder for the PNG images.")
