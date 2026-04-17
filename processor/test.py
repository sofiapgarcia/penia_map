from data_preparation import analyze_excel_file , encode_categorical_columns
from data_visualization import plot_distribution
from feature_reduction import *
from ml_pipeline import *
from data_augmentation import *
import pandas as pd

#filepath = r"C:\Users\Sofia\Desktop\Tese\dados\dataset.xlsx"
"""
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
"""
#df_reduced.to_excel(r"C:\Users\Sofia\Desktop\Tese\df_reduced.xlsx")


filepath2 = r"C:\Users\Sofia\Desktop\Tese\df_reduced.xlsx"

# read the data
df_reduced = pd.read_excel(filepath2)

# apply data augmentation

# SMOGN
df_smogn = apply_smogn(df_reduced, k=2, samp_method="extreme", preserve_original=True)
# verify the distribution of the augmented dataset
#plot_distribution(df_smogn, "Estimated metabolic age", bin_width=1)

# Gaussian Copula
df_gc = augment_dataframe_gaussian_copula(df_reduced, n_samples=30)
# verify the distribution of the augmented dataset
#plot_distribution(df_gc, "Estimated metabolic age", bin_width=1)

# Gaussian Copula + SMOGN
df_smogn_gc = augment_dataframe_gaussian_copula(df_smogn, n_samples=20)
plot_distribution(df_smogn_gc, "Estimated metabolic age", bin_width=1)
# Additional functions

# plot distribution of one variable
#plot_distribution(df_features, "Estimated metabolic age")
print("oi")
"""
# Principal Component Analysis
# decide the suitable number of principal components
pca_full, cumulative = inspect_pca_variance(df_reduced)
n_components = 7
# apply pca
pca, X_pca_df, loadings = apply_pca(df_reduced, n_components)



"""