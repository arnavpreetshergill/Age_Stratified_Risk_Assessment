import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load your dataset
# Replace 'your_dataset.csv' with the path to your file.
# For this example, we will load a built-in dataset if a file isn't provided.
try:
    df = pd.read_csv('heart_data_elderly.csv')
    print("User dataset loaded.")
except FileNotFoundError:
    print("Example dataset loaded (Iris) for demonstration.")
    df = sns.load_dataset('iris')

# 2. Pre-processing
# Correlation requires numerical data. We drop non-numeric columns automatically.
numerical_df = df.select_dtypes(include=['float64', 'int64'])

# 3. Calculate the Correlation Matrix
# method='pearson' is standard. Options: 'kendall', 'spearman'
corr_matrix = numerical_df.corr()

# 4. Visualization (Heatmap)
plt.figure(figsize=(10, 8))

# Create a mask to hide the upper triangle (optional, but makes it cleaner)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, 
            annot=True,         # Write the data value in each cell
            fmt=".2f",          # Round to 2 decimal places
            cmap='coolwarm',    # Color map: Red (positive), Blue (negative)
            vmin=-1, vmax=1,    # Set scale limits
            center=0,           # Set center of color map to 0
            square=True,        # Force square cells
            mask=mask)          # Apply the mask (remove this line to see full matrix)

plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()
