import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = 'processed_train_100k_stratified.csv'
RAW_FILE = 'heart_statlog_cleveland_hungary_final(1).csv'


def get_z_cutoffs(raw_file):
    raw_df = pd.read_csv(raw_file)
    mean_age = raw_df['age'].mean()
    std_age = raw_df['age'].std()
    z45 = (45 - mean_age) / std_age
    z65 = (65 - mean_age) / std_age
    return z45, z65


def age_group_from_z(age_z, z45, z65):
    if age_z < z45:
        return "Young"
    if age_z <= z65:
        return "Middle"
    return "Elderly"


def main():
    df = pd.read_csv(DATA_FILE)
    z45, z65 = get_z_cutoffs(RAW_FILE)
    print(df.head())
    print(f"Using z-cutoffs: 45y={z45:.3f}, 65y={z65:.3f}")

    df["age_group"] = df["age"].apply(lambda x: age_group_from_z(x, z45, z65))
    sns.countplot(x="age_group", hue="target", data=df)
    plt.title("Heart Disease Distribution Across Age Groups")
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("age_group_distribution.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
