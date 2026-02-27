# SMOTE vs Non-SMOTE Cardiovascular Disease Comparison (Cohort-Tuned GridSearch)

Target label meaning: `1 = cardiovascular disease`, `0 = no cardiovascular disease`.

## Cohort rules

- `Young`: age < 45
- `Middle`: 45 <= age <= 65
- `Elderly`: age > 65

## GridSearch configuration

- Method: GridSearchCV
- Scoring: f1
- Param grid: {'n_estimators': [80, 140], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.15], 'subsample': [0.8, 1.0]}
- Grid size per cohort: 16

## Overall train class distribution

- Without SMOTE (before): {0: 328, 1: 405}
- Without SMOTE (after):  {0: 328, 1: 405}
- With SMOTE (before):    {0: 328, 1: 405}
- With SMOTE (after):     {0: 442, 1: 442}

## Overall performance summary

| Metric | Without SMOTE | With SMOTE | Delta (With - Without) | Better |
|---|---:|---:|---:|---|
| Accuracy | 0.8370 | 0.8478 | +0.0109 | With SMOTE |
| Precision | 0.8600 | 0.8854 | +0.0254 | With SMOTE |
| Recall | 0.8431 | 0.8333 | -0.0098 | Without SMOTE |
| F1 | 0.8515 | 0.8586 | +0.0071 | With SMOTE |
| ROC-AUC | 0.9108 | 0.9116 | +0.0008 | With SMOTE |
| PR-AUC | 0.9302 | 0.9298 | -0.0003 | Without SMOTE |

## Per-cohort performance summary

| Cohort | Metric | Without SMOTE | With SMOTE | Delta (With - Without) | Better |
|---|---|---:|---:|---:|---|
| Young | Accuracy | 0.8718 | 0.8974 | +0.0256 | With SMOTE |
| Young | Recall | 0.8571 | 1.0000 | +0.1429 | With SMOTE |
| Young | F1 | 0.7059 | 0.7778 | +0.0719 | With SMOTE |
| Young | ROC-AUC | 0.9598 | 0.9598 | -0.0000 | Without SMOTE |
| Young | PR-AUC | 0.8320 | 0.8084 | -0.0236 | Without SMOTE |
| Middle | Accuracy | 0.8235 | 0.8309 | +0.0074 | With SMOTE |
| Middle | Recall | 0.8391 | 0.8161 | -0.0230 | Without SMOTE |
| Middle | F1 | 0.8588 | 0.8606 | +0.0018 | With SMOTE |
| Middle | ROC-AUC | 0.9001 | 0.9059 | +0.0059 | With SMOTE |
| Middle | PR-AUC | 0.9434 | 0.9469 | +0.0034 | With SMOTE |
| Elderly | Accuracy | 0.8889 | 0.8889 | +0.0000 | Tie |
| Elderly | Recall | 0.8750 | 0.8750 | +0.0000 | Tie |
| Elderly | F1 | 0.9333 | 0.9333 | +0.0000 | Tie |
| Elderly | ROC-AUC | 1.0000 | 1.0000 | +0.0000 | Tie |
| Elderly | PR-AUC | 1.0000 | 1.0000 | +0.0000 | Tie |

## Best hyperparameters by cohort

| Cohort | Without SMOTE | With SMOTE |
|---|---|---|
| Young | n_estimators=80, max_depth=5, learning_rate=0.15, subsample=1.0 | n_estimators=80, max_depth=3, learning_rate=0.05, subsample=1.0 |
| Middle | n_estimators=140, max_depth=5, learning_rate=0.15, subsample=1.0 | n_estimators=80, max_depth=5, learning_rate=0.15, subsample=0.8 |
| Elderly | n_estimators=80, max_depth=5, learning_rate=0.05, subsample=0.8 | n_estimators=140, max_depth=5, learning_rate=0.05, subsample=0.8 |

## Overall confusion matrices

- Without SMOTE [TN FP FN TP]: [68 14 16 86]
- With SMOTE [TN FP FN TP]: [71 11 17 85]
