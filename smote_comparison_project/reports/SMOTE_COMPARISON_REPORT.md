# SMOTE vs Non-SMOTE Cardiovascular Disease Comparison (Fixed Train/Test Holdout)

Target label meaning: `1 = cardiovascular disease`, `0 = no cardiovascular disease`.

## Data sources

- Raw dataset: `C:\Users\Admin\Desktop\mern\DM_Project\datasets\raw\heart_statlog_cleveland_hungary_final(1).csv`
- Evaluation mode: `fixed_generated_train_with_real_holdout_test`
- Generated train file: `C:\Users\Admin\Desktop\mern\DM_Project\datasets\processed\processed_train_100k_stratified.csv`
- Processed test file: `C:\Users\Admin\Desktop\mern\DM_Project\datasets\processed\processed_test.csv`
- Generated train rows: `100000`
- Test source: fixed processed_test.csv transformed with the train-fitted preprocessor
- Processed age cutoffs: derived once from the single train-fitted scaler used for processed_train/processed_test

## Cohort rules

- `Young`: age < 45
- `Middle`: 45 <= age <= 65
- `Elderly`: age > 65

## Model parameter configuration

- Method: default_xgboost_params
- Selection policy: fixed_default_values
- Params: {'n_estimators': 120, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 1.0, 'colsample_bytree': 1.0, 'eval_metric': 'logloss', 'n_jobs': 1}

## Feature selection configuration

- Method: xgboost_feature_importance_top_k
- Top K: 10
- Stage: before the single holdout fit on each cohort

## Overall train class distribution

- Without SMOTE (before): {0: 44774, 1: 55226}
- Without SMOTE (after):  {0: 44774, 1: 55226}
- With SMOTE (before):    {0: 44774, 1: 55226}
- With SMOTE (after):     {0: 60375, 1: 60375}

## Overall performance summary

| Metric | Without SMOTE | With SMOTE | Delta (With - Without) | Better |
|---|---:|---:|---:|---|
| Accuracy | 0.8514 | 0.8587 | +0.0072 | With SMOTE |
| Precision | 0.8590 | 0.8608 | +0.0018 | With SMOTE |
| Recall | 0.8758 | 0.8889 | +0.0131 | With SMOTE |
| F1 | 0.8673 | 0.8746 | +0.0073 | With SMOTE |
| ROC-AUC | 0.8989 | 0.8963 | -0.0026 | Without SMOTE |
| PR-AUC | 0.9040 | 0.8942 | -0.0098 | Without SMOTE |

## Per-cohort performance summary

| Cohort | Metric | Without SMOTE | With SMOTE | Delta (With - Without) | Better |
|---|---|---:|---:|---:|---|
| Young | Accuracy | 0.8182 | 0.8182 | +0.0000 | Tie |
| Young | Recall | 0.7692 | 0.7692 | +0.0000 | Tie |
| Young | F1 | 0.6667 | 0.6667 | +0.0000 | Tie |
| Young | ROC-AUC | 0.8810 | 0.8938 | +0.0128 | With SMOTE |
| Young | PR-AUC | 0.7496 | 0.7781 | +0.0285 | With SMOTE |
| Middle | Accuracy | 0.8571 | 0.8719 | +0.0148 | With SMOTE |
| Middle | Recall | 0.8819 | 0.8976 | +0.0157 | With SMOTE |
| Middle | F1 | 0.8854 | 0.8976 | +0.0123 | With SMOTE |
| Middle | ROC-AUC | 0.9047 | 0.9013 | -0.0034 | Without SMOTE |
| Middle | PR-AUC | 0.9283 | 0.9205 | -0.0079 | Without SMOTE |
| Elderly | Accuracy | 0.8889 | 0.8333 | -0.0556 | Without SMOTE |
| Elderly | Recall | 0.9231 | 0.9231 | +0.0000 | Tie |
| Elderly | F1 | 0.9231 | 0.8889 | -0.0342 | Without SMOTE |
| Elderly | ROC-AUC | 0.9077 | 0.9231 | +0.0154 | With SMOTE |
| Elderly | PR-AUC | 0.9727 | 0.9760 | +0.0033 | With SMOTE |

## Evaluation split details

- Generated training rows: `100000`
- Test rows: `276`
- Without SMOTE train rows after sampling: `100000`
- With SMOTE train rows after sampling: `120750`

## Model parameters by cohort

| Cohort | Without SMOTE | With SMOTE |
|---|---|---|
| Young | n_estimators=120, max_depth=4, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, eval_metric=logloss, n_jobs=1 | n_estimators=120, max_depth=4, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, eval_metric=logloss, n_jobs=1 |
| Middle | n_estimators=120, max_depth=4, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, eval_metric=logloss, n_jobs=1 | n_estimators=120, max_depth=4, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, eval_metric=logloss, n_jobs=1 |
| Elderly | n_estimators=120, max_depth=4, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, eval_metric=logloss, n_jobs=1 | n_estimators=120, max_depth=4, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, eval_metric=logloss, n_jobs=1 |

## Selected top features by cohort

| Cohort | Without SMOTE | With SMOTE |
|---|---|---|
| Young | chest_pain_type_4, exercise_angina, st_slope_2, fasting_blood_sugar, oldpeak, chest_pain_type_3, resting_ecg_2, max_heart_rate, chest_pain_type_2, sex | chest_pain_type_4, exercise_angina, st_slope_2, fasting_blood_sugar, oldpeak, chest_pain_type_3, resting_ecg_2, max_heart_rate, chest_pain_type_2, sex |
| Middle | st_slope_2, exercise_angina, st_slope_3, chest_pain_type_4, fasting_blood_sugar, sex, cholesterol, oldpeak, chest_pain_type_2, max_heart_rate | st_slope_2, exercise_angina, st_slope_3, chest_pain_type_4, fasting_blood_sugar, sex, cholesterol, oldpeak, chest_pain_type_2, max_heart_rate |
| Elderly | chest_pain_type_4, sex, exercise_angina, st_slope_2, resting_ecg_2, fasting_blood_sugar, chest_pain_type_3, resting_ecg_1, oldpeak, resting_bp_s | chest_pain_type_4, sex, exercise_angina, st_slope_2, resting_ecg_2, fasting_blood_sugar, chest_pain_type_3, resting_ecg_1, oldpeak, resting_bp_s |

## Overall confusion matrices

- Without SMOTE [TN FP FN TP]: [101 22 19 134]
- With SMOTE [TN FP FN TP]: [101 22 17 136]

## Visualization outputs

- `overall_metrics_comparison.png`
- `per_cohort_metrics_comparison.png`
- `confusion_matrix_comparison.png`
- `sampling_comparison.png`
- `feature_selection_comparison.png`
