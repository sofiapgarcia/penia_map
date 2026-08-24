from ml_pipeline.preprocessing.data_augmentation import *
from ml_pipeline.preprocessing.feature_reduction import select_features_by_group , adaptive_correlation_matrix , top_correlations_per_target
from ml_pipeline.preprocessing.linear_regression_feature_reduction import run_for_multiple_targets , summarize_results , get_selected_features_dict , results_to_dataframe
from ml_pipeline.modeling.test_models import run_training_pipeline , split_train_test_df , evaluate_production_model_simple , evaluate_production_model_k_fold
from constants import RANDOM_SEED , best_models_all_variables , best_models_penia_map
import pandas as pd

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"

# read the data
df = pd.read_excel(filepath)

# remove participant column ID
df_features = df.drop(columns=["ID"], errors="ignore")

ALL_VARIABLES = False

features_dict = best_models_all_variables if ALL_VARIABLES else best_models_penia_map

target = "Age (y)"
#target = "Estimated metabolic age"
#target = "Bone Mineral Density"
#target = "peak_power"
features = features_dict[target] + [target]

df_train , df_test = split_train_test_df(df_features, target_col= target, test_size=0.15 , random_state = 3)

df_train = df_train[features]
df_test = df_test[features]
df_reduced = df_features[features]

# apply data augmentation
# Gaussian Copula + SMOGN
#df_train = apply_data_augmentation(df_train, n_samples = 20)

# Model selection
#results_summary = run_training_pipeline(df_train, splits_inner=3, splits_outer=5, target_column=target)


suffix = "all_variables" if ALL_VARIABLES else "penia_map"

#filepath3 = rf"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\results_{target}_{suffix}.xlsx"

#results_summary.to_excel(filepath3, index=False)


# Evaluation of production model
#evaluate_production_model_simple(df_train, df_test, model_name="Random Forest",target_column=target)
results = evaluate_production_model_k_fold(df_reduced, model_name="KNN",target_column=target, normalization="minmax")
