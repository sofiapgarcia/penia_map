from ml_pipeline.preprocessing.data_augmentation import *
from ml_pipeline.preprocessing.feature_reduction import select_features_by_group , adaptive_correlation_matrix , top_correlations_per_target
from ml_pipeline.preprocessing.linear_regression_feature_reduction import run_for_multiple_targets ,  results_to_dataframe
from ml_pipeline.modeling.test_models import run_training_pipeline , split_train_test_df , evaluate_production_model_simple , evaluate_production_model_k_fold
from constants import RANDOM_SEED , best_models_all_variables , best_models_penia_map
import pandas as pd

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe_filtered.xlsx"

# read the data
df = pd.read_excel(filepath)

# remove participant column ID
df_features = df.drop(columns=["ID","Torso muscle kg"], errors="ignore")


cor_matrix , corr_pvalues , method_matrix = adaptive_correlation_matrix(df_features)

targets = {"Estimated metabolic age": ["Age (y)"],
"peak_power": ["velocity_at_peak_power", "force_at_peak_power", "peak_power_mass", "peak_power_height_squared", "max_force", "max_velocity","rep_pp_hd_peak_rfd","mean_hd_peak_rfd"],
"Bone Mineral Density": ["Estimated bone mass kg"],
           "Age (y)":["Estimated metabolic age"]
}

corr_path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\correlations.xlsx"

result_df = top_correlations_per_target(targets=targets,corr_df=cor_matrix,pval_df=corr_pvalues,method_df=method_matrix,top_n=30,excel_path=corr_path)
