import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# --- CONFIGURATION ---
# List of your split files and their friendly names
DATASETS = [
    {'file': 'heart_data_young.csv', 'name': 'Young Adults (<45)'},
    {'file': 'heart_data_middle.csv', 'name': 'Middle-Aged (45-65)'},
    {'file': 'heart_data_elderly.csv', 'name': 'Elderly (>65)'}
]
RANDOM_STATE = 42

def analyze_group(file_path, group_name):
    print("="*60)
    print(f"STARTING ANALYSIS: {group_name}")
    print("="*60)

    # 1. Load Data
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found. Skipping.")
        return

    df = pd.read_csv(file_path)
    print(f"Data Loaded: {len(df)} rows")
    if len(df) < 10 or df['target'].nunique() < 2:
        print(f"Skipping {group_name}: not enough rows/classes for train/test split.")
        return

    # 2. Prepare X and y
    target_col = 'target'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 4. Train Gradient Boosting Model (XGBoost)
    # We use XGBoost because it is the "Gold Standard" for tabular data and SHAP
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)

    # 5. Evaluate Performance
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy for {group_name}: {acc:.2%}")
    print("-" * 30)

    # 6. SHAP Analysis
    print(f"Generating SHAP plots for {group_name}...")
    
    # Create object that can calculate shap values
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    # --- PLOT 1: Summary Plot (The "Dot" Plot) ---
    # Shows direction of effect (Red = High Feature Value => High/Low Risk)
    plt.figure(figsize=(10, 6))
    plt.title(f"SHAP Summary: {group_name}")
    shap.summary_plot(shap_values, X_test, show=False)
    
    # Save plot
    safe_name = group_name.split()[0] # e.g., "Young"
    plot_filename = f"shap_summary_{safe_name}.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_filename}")

    # --- PLOT 2: Bar Plot (Feature Importance) ---
    # Shows absolute importance (Magnitude only)
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance: {group_name}")
    shap.plots.bar(shap_values, show=False)
    
    bar_filename = f"shap_importance_{safe_name}.png"
    plt.savefig(bar_filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: {bar_filename}")
    print("\n")

# --- EXECUTION LOOP ---
if __name__ == "__main__":
    for data in DATASETS:
        analyze_group(data['file'], data['name'])
    
    print("Done! Check your folder for the PNG images.")
