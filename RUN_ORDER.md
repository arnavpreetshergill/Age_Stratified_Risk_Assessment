# Run Order

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
- `processed_heart_data_full.csv`
- `processed_train.csv`
- `processed_test.csv`

## 3) Augment train split only
This creates synthetic samples from `processed_train.csv` and keeps test data untouched.

```powershell
py dataGeneration.py
```

Generated file:
- `processed_train_100k_stratified.csv`

## 4) Split augmented train by age group
Required for per-group SHAP analysis scripts.

```powershell
py data_split.py
```

Generated files:
- `heart_data_young.csv`
- `heart_data_middle.csv`
- `heart_data_elderly.csv`

## 5) Compare baseline vs age-stratified models
This script now runs a leakage-safe evaluation pipeline and reports:
- Accuracy
- ROC-AUC
- PR-AUC
- Recall
- F1
- Confusion matrices

```powershell
py performance_comparison.py
```

## 6) Generate SHAP explainability plots

```powershell
py gradient_boost_SHAP.py
```

Generated files:
- `shap_summary_Young.png`
- `shap_summary_Middle-Aged.png`
- `shap_summary_Elderly.png`
- `shap_importance_Young.png`
- `shap_importance_Middle-Aged.png`
- `shap_importance_Elderly.png`

## 7) Optional visualizations

```powershell
py vis.py
py correlation.py
```

Notes:
- `vis.py` saves `age_group_distribution.png`.
- `correlation.py` shows a heatmap window.
