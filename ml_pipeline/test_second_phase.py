from ml_pipeline.preprocessing.data_augmentation import *
from ml_pipeline.preprocessing.feature_reduction import select_features_by_group
from ml_pipeline.modeling.test_models import run_training_pipeline , split_train_test_df , evaluate_production_model_simple , evaluate_production_model_k_fold
from constants import RANDOM_SEED
import pandas as pd

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"

# read the data
df = pd.read_excel(filepath)
"""
# identify missing values, duplicates and outliers
#df = analyze_excel_file(filepath)

# encode categorical columns
df_encoded = encode_categorical_columns(df)
"""
# remove participant column ID
df_features = df.drop(columns=["ID"], errors="ignore")

# get correlation by each dimension of the penia map
#corr_penia = correlation_by_groups(df_features)

# reduce the number of features based on correlation by group
df_reduced = select_features_by_group(df_features, selected_groups=["Osteopenia", "Powerpenia", "Dynapenia", "Sarcopenia"] , n_features_per_group=3 , verbose=True)


df_train , df_test = split_train_test_df(df_reduced, test_size=0.05 , random_state = 3)

# apply data augmentation
# Gaussian Copula + SMOGN
df_train = apply_data_augmentation(df_reduced, n_samples = 20)

# analyse the new distribution of target
#plot_distribution(df_augmented, "Estimated metabolic age", bin_width=1)

# Model selection
results_summary = run_training_pipeline(df_train, splits_inner=3, splits_outer=5)


filepath3 = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\results_summary_second_phase.xlsx"

results_summary.to_excel(filepath3, index=False)

# Evaluation of production model
#evaluate_production_model_simple(df_train, df_test)
#evaluate_production_model_k_fold(df_reduced)


# Additional functions

# plot distribution of one variable
#plot_distribution(df_features, "Estimated metabolic age")
