# SMOTE vs Non-SMOTE Cardiovascular Disease Model Comparison

Target label meaning: `1 = cardiovascular disease`, `0 = no cardiovascular disease`.

## Train class distribution

- Without SMOTE (before): {0: 328, 1: 405}
- Without SMOTE (after):  {0: 328, 1: 405}
- With SMOTE (before):    {0: 328, 1: 405}
- With SMOTE (after):     {0: 405, 1: 405}

## Performance summary

| Metric | Without SMOTE | With SMOTE | Delta (With - Without) | Better |
|---|---:|---:|---:|---|
| Accuracy | 0.8750 | 0.8641 | -0.0109 | Without SMOTE |
| Precision | 0.8911 | 0.8812 | -0.0099 | Without SMOTE |
| Recall | 0.8824 | 0.8725 | -0.0098 | Without SMOTE |
| F1 | 0.8867 | 0.8768 | -0.0099 | Without SMOTE |
| ROC-AUC | 0.9242 | 0.9219 | -0.0023 | Without SMOTE |
| PR-AUC | 0.9309 | 0.9271 | -0.0037 | Without SMOTE |

## Confusion matrices

- Without SMOTE [TN FP FN TP]: [71 11 12 90]
- With SMOTE [TN FP FN TP]: [70 12 13 89]
