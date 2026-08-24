from ml_pipeline.preprocessing.data_augmentation import *
from ml_pipeline.preprocessing.feature_reduction import select_features_by_group , adaptive_correlation_matrix , top_correlations_per_target
from ml_pipeline.preprocessing.linear_regression_feature_reduction import run_for_multiple_targets , summarize_results , get_selected_features_dict , results_to_dataframe
from ml_pipeline.modeling.test_models import run_training_pipeline , split_train_test_df , evaluate_production_model_simple , evaluate_production_model_k_fold
from constants import RANDOM_SEED , best_models_all_variables , best_models_penia_map
import pandas as pd

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\penia_map_pca_peak_power.xlsx"

# read the data
df_features = pd.read_excel(filepath)

#target = "Age (y)"
#target = "Estimated metabolic age"
#target = "Bone Mineral Density"
target = "peak_power"

#df_features = df.drop(columns=["Age (y)"], errors="ignore")

df_train , df_test = split_train_test_df(df_features, target_col= target, test_size=0.15 , random_state = 3)

# Model selection
results_summary = run_training_pipeline(df_train, splits_inner=3, splits_outer=5, target_column=target)


filepath3 = rf"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\results_{target}_pca.xlsx"

results_summary.to_excel(filepath3, index=False)


# Evaluation of production model
#evaluate_production_model_simple(df_train, df_test, model_name="Random Forest",target_column=target)
#results = evaluate_production_model_k_fold(df_reduced, model_name="KNN",target_column=target, normalization="minmax")