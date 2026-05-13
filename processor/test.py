from data_preparation import analyze_excel_file , encode_categorical_columns
from data_visualization import plot_distribution
from feature_reduction import *
from ml_pipeline import *
from data_augmentation import *
from test_models import run_training_pipeline , evaluate_production_model , split_train_test_df
import pandas as pd

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\dataset_tanita_corrected.xlsx"
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
#corr_penia = correlation_by_groups(df_features)

# reduce the number of features based on correlation by group
df_reduced = select_features_by_group(df_features,selected_groups=["Osteopenia", "Powerpenia", "Dynapenia", "Sarcopenia", "Body Fat"],n_features_per_group=1 , verbose=True)
"""
filepath2 = r"C:\Users\Sofia\Desktop\Tese\dataset_reduced\df_reduced_tanita_corrected_1feature_without_outlier.xlsx"

#df_reduced.to_excel(filepath2)

# read the data
df_reduced = pd.read_excel(filepath2, index_col=0)

df_train , df_test = split_train_test_df(df_reduced, test_size=0.15)

# apply data augmentation
# Gaussian Copula + SMOGN
df_augmented = apply_data_augmentation(df_train)

# analyse the new distribution of target
#plot_distribution(df_augmented, "Estimated metabolic age", bin_width=1)

# Model selection
results_summary = run_training_pipeline(df_augmented, splits_inner=3, splits_outer=5)

filepath3 = r"C:\Users\Sofia\Desktop\Tese\results_summary.xlsx"

results_summary.to_excel(filepath3, index=False)

# Evaluation of production model
#evaluate_production_model(df_train, df_test)


# Additional functions

# plot distribution of one variable
#plot_distribution(df_features, "Estimated metabolic age")
print("oi")