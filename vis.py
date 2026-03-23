import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from preprocess import NUMERIC_FEATURES, TARGET_COL, build_preprocessor, load_and_clean_data
from project_paths import (
    AGE_GROUP_DISTRIBUTION_FILE,
    PROCESSED_TRAIN_100K_FILE,
    RAW_DATA_FILE,
    ensure_root_artifact_dirs,
)

DATA_FILE = PROCESSED_TRAIN_100K_FILE
RAW_FILE = RAW_DATA_FILE
OUTPUT_FILE = AGE_GROUP_DISTRIBUTION_FILE


def get_z_cutoffs(raw_file):
    raw_df = load_and_clean_data(raw_file)
    X_raw = raw_df.drop(columns=[TARGET_COL])
    y_raw = raw_df[TARGET_COL].astype(int)
    X_train_raw, _, _, _ = train_test_split(
        X_raw,
        y_raw,
        test_size=0.2,
        random_state=42,
        stratify=y_raw,
    )
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train_raw)
    scaler = preprocessor.named_transformers_["num"]
    age_idx = NUMERIC_FEATURES.index("age")
    mean_age = float(scaler.mean_[age_idx])
    std_age = float(scaler.scale_[age_idx])
    z45 = (45 - mean_age) / std_age
    z65 = (65 - mean_age) / std_age
    return z45, z65, mean_age, std_age


def age_group_from_z(age_z, z45, z65):
    if age_z < z45:
        return "Young"
    if age_z <= z65:
        return "Middle"
    return "Elderly"


def main():
    df = pd.read_csv(DATA_FILE)
    z45, z65, mean_age, std_age = get_z_cutoffs(RAW_FILE)
    print(df.head())
    print(
        f"Using train-split z-cutoffs: 45y={z45:.3f}, 65y={z65:.3f} "
        f"(mean={mean_age:.3f}, std={std_age:.3f})"
    )

    df["age_group"] = df["age"].apply(lambda x: age_group_from_z(x, z45, z65))
    sns.countplot(x="age_group", hue="target", data=df)
    plt.title("Heart Disease Distribution Across Age Groups")
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=200)
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    ensure_root_artifact_dirs()
    main()
