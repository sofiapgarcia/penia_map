from data_preparation import analyze_excel_file , encode_categorical_columns
from data_visualization import plot_distribution
from feature_reduction import *
from ml_pipeline import *
import pandas as pd

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\dataset.xlsx"

# read the data

df = pd.read_excel(filepath)

# identify missing values, duplicates and outliers
#df = analyze_excel_file(filepath)

# encode categorical columns
df_encoded = encode_categorical_columns(df)

# remove participant column ID
df_features = df_encoded.drop(columns=["ID"], errors="ignore")

# get correlation by each dimension of the penia map
corr_penia = correlation_by_groups(df_features)

# reduce the number of features based on correlation by group
df_reduced = select_top_features_by_group(df_features,selected_groups=["Osteopenia", "Powerpenia", "Dynapenia", "Sarcopenia", "Body Fat"],verbose=False)

target_column = "Estimated metabolic age"
print(df_reduced[target_column].nunique())
print(df_reduced[target_column].describe())
print(df_reduced[target_column].value_counts().head(10))

import numpy as np

q1 = np.percentile(df_reduced[target_column], 25)
q3 = np.percentile(df_reduced[target_column], 75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

print(lower, upper)

# apply data augmentation
df_augmented = apply_smogn(df_reduced)

"""
# plot distribution of one variable
#plot_distribution(df_features, "Estimated metabolic age")

# Principal Component Analysis
# decide the suitable number of principal components
pca_full, cumulative = inspect_pca_variance(df_reduced)
n_components = 7
# apply pca
pca, X_pca_df, loadings = apply_pca(df_reduced, n_components)


print("oi")
"""