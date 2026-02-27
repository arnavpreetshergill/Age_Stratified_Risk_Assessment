# Run Order - SMOTE vs Non-SMOTE Project

## 1) Go to the new comparison project

```powershell
cd c:\Users\Admin\Desktop\mern\DM_Project\smote_comparison_project
```

## 2) Install dependencies (if not already installed)

```powershell
py -m pip install -r requirements.txt
```

## 3) Run both variants and compare metrics

```powershell
py compare_smote_vs_no_smote.py
```

This command will:
- Train and evaluate the `without_smote` variant
- Train and evaluate the `with_smote` variant
- Save per-variant JSON outputs:
  - `without_smote\results_without_smote.json`
  - `with_smote\results_with_smote.json`
- Save the markdown comparison report:
  - `SMOTE_COMPARISON_REPORT.md`

## 4) (Optional) Run each variant independently

Without SMOTE:

```powershell
py without_smote\run_without_smote.py
```

With SMOTE:

```powershell
py with_smote\run_with_smote.py
```
