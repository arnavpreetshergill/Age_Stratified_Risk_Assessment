# Run Order

## 0) Run the full pipeline once
This executes the entire root pipeline plus the SMOTE comparison subproject in order.

```powershell
py run_all.py
```

Main artifact folders created:
- `datasets\`
- `visualizations\`
- `smote_comparison_project\results\`
- `smote_comparison_project\reports\`
- `smote_comparison_project\visualizations\`

## 1) Install dependencies
Use the Python launcher (`py`) on Windows:

```powershell
py -m pip install -r requirements.txt
```

## 2) Preprocess raw data
This creates a full processed dataset plus train/test processed splits (no preprocessing leakage).

```powershell
py preprocess.py
```

Generated files:
- `datasets\processed\processed_heart_data_full.csv`
- `datasets\processed\processed_train.csv`
- `datasets\processed\processed_test.csv`

## 3) Augment train split only
This creates synthetic samples from `processed_train.csv` and keeps test data untouched.

```powershell
py dataGeneration.py
```

Generated file:
- `datasets\processed\processed_train_100k_stratified.csv`

## 4) Split augmented train by age group
Optional. Use this only if you want exported cohort CSV files from the augmented training set.

```powershell
py data_split.py
```

Generated files:
- `datasets\cohorts\heart_data_young.csv`
- `datasets\cohorts\heart_data_middle.csv`
- `datasets\cohorts\heart_data_elderly.csv`

## 5) Compare baseline vs age-stratified models
This script now runs a leakage-safe evaluation pipeline and reports:
- Accuracy
- ROC-AUC
- PR-AUC
- Recall
- F1
- Confusion matrices
- Top-10 feature selection on training data only
- Train-only SMOTE after feature selection
- No 100k train augmentation inside this script
- Visualization PNGs in `visualizations\performance\`

Note:
- This step uses the normal train/test split from `preprocess.py`.
- Steps 3 and 4 are not required for this script.

```powershell
py performance_comparison.py
```

Generated visualization files:
- `visualizations\performance\overall_metrics_comparison.png`
- `visualizations\performance\per_cohort_metrics_comparison.png`
- `visualizations\performance\confusion_matrix_comparison.png`
- `visualizations\performance\sampling_comparison.png`
- `visualizations\performance\feature_selection_comparison.png`

## 6) Generate SHAP explainability plots
This now uses the same clean train/test split as `performance_comparison.py`, then trains the cohort models with the same top-10 feature selection + train-only SMOTE flow before generating SHAP plots.

```powershell
py gradient_boost_SHAP.py
```

Generated files:
- `visualizations\shap\shap_summary_Young.png`
- `visualizations\shap\shap_summary_Middle-Aged.png`
- `visualizations\shap\shap_summary_Elderly.png`
- `visualizations\shap\shap_importance_Young.png`
- `visualizations\shap\shap_importance_Middle-Aged.png`
- `visualizations\shap\shap_importance_Elderly.png`

## 7) Optional visualizations

```powershell
py vis.py
py correlation.py
```

Notes:
- `vis.py` saves `visualizations\exploratory\age_group_distribution.png`.
- `correlation.py` saves `visualizations\exploratory\correlation_heatmap.png`.
