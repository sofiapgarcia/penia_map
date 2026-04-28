from data_preparation import analyze_excel_file , encode_categorical_columns
from data_visualization import plot_distribution
from feature_reduction import *
from ml_pipeline import *
from data_augmentation import *
from test_models import run_training_pipeline , evaluate_production_model
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
df_reduced = select_features_by_group(df_features,selected_groups=["Osteopenia", "Powerpenia", "Dynapenia", "Sarcopenia", "Body Fat"],verbose=True)

#df_reduced.to_excel(r"C:\Users\Sofia\Desktop\Tese\df_reduced_with_fat.xlsx")



filepath2 = r"C:\Users\Sofia\Desktop\Tese\df_reduced.xlsx"
"""
# read the data
df_reduced = pd.read_excel(filepath2, index_col=0)

# apply data augmentation

# Gaussian Copula + SMOGN
df_augmented = apply_data_augmentation(df_reduced)
# analyse the new distribution of target
#plot_distribution(df_augmented, "Estimated metabolic age", bin_width=1)

# Model selection
X_train, X_test, y_train, y_test = run_training_pipeline(df_augmented, splits_inner=2, splits_outer=5, models_to_run=["LR"])

# Evaluation of production model
#evaluate_production_model(X_train, X_test, y_train, y_test)


# Additional functions

# plot distribution of one variable
#plot_distribution(df_features, "Estimated metabolic age")
print("oi")

# Principal Component Analysis
# decide the suitable number of principal components
pca_full, cumulative = inspect_pca_variance(df_reduced)
n_components = 7
# apply pca
pca, X_pca_df, loadings = apply_pca(df_reduced, n_components)



"""