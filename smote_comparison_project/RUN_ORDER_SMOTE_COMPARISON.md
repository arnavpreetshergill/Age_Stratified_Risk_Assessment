# Run Order - SMOTE vs Non-SMOTE Project

## 1) Go to the new comparison project

```powershell
cd c:\Users\Admin\Desktop\mern\DM_Project\smote_comparison_project
```

## 2) Install dependencies (if not already installed)

```powershell
py -m pip install -r requirements.txt
```

## 3) Make sure the generated datasets already exist in the parent project

Required files in `c:\Users\Admin\Desktop\mern\DM_Project`:

- `datasets\processed\processed_train_100k_stratified.csv`
- `datasets\processed\processed_test.csv`
- `datasets\raw\heart_statlog_cleveland_hungary_final(1).csv`

These are used as:
- synthetic processed training dataset
- processed holdout test dataset
- raw reference dataset to recover the processed-space age cutoffs

## 4) Run both variants and compare metrics

```powershell
py compare_smote_vs_no_smote.py
```

This command will:
- Train and evaluate the `without_smote` variant
- Train and evaluate the `with_smote` variant
- Train separate models for `Young`, `Middle`, and `Elderly` cohorts
- Select the top 10 training-only features inside each cohort before model fitting
- Run `GridSearchCV` hyperparameter tuning independently per cohort
- Save per-variant JSON outputs:
  - `results\without_smote\results_without_smote.json`
  - `results\with_smote\results_with_smote.json`
- Save the markdown comparison report:
  - `reports\SMOTE_COMPARISON_REPORT.md`
- Save visualization PNGs in:
  - `visualizations\overall_metrics_comparison.png`
  - `visualizations\per_cohort_metrics_comparison.png`
  - `visualizations\confusion_matrix_comparison.png`
  - `visualizations\sampling_comparison.png`
  - `visualizations\feature_selection_comparison.png`
- Use the artificial processed training dataset instead of retraining from the raw CSV split inside this subproject

Note:
- Runtime is longer than fixed-parameter training because each cohort performs cross-validated grid search.

## 5) (Optional) Run each variant independently

Without SMOTE:

```powershell
py without_smote\run_without_smote.py
```

With SMOTE:

```powershell
py with_smote\run_with_smote.py
```
