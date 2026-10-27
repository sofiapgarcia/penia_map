from ml_pipeline.preprocessing.linear_regression_feature_reduction import adaptive_correlation_matrix , top_correlations_per_target
from ml_pipeline.preprocessing.linear_regression_feature_reduction import run_for_multiple_targets ,  _results_to_dataframe
from constants import RANDOM_SEED , best_models_all_variables , best_models_penia_map
import pandas as pd

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe_filtered_without_mean_variables.xlsx"

# read the data
df = pd.read_excel(filepath)

# remove participant column ID
df_features = df.drop(columns=["ID","Torso muscle kg"], errors="ignore")
target = "Age (y)"
df_train , df_test = split_train_test_df(df_features, target_col= target, test_size=0.15 , random_state = 42)
path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\train_age.xlsx"
df_train.to_excel(path)


cor_matrix , corr_pvalues , method_matrix = adaptive_correlation_matrix(df_train)

targets = {"Estimated metabolic age": ["Age (y)"],
"peak_power": ["velocity_at_peak_power", "force_at_peak_power", "peak_power_mass", "peak_power_height_squared", "max_force", "max_velocity","rep_pp_hd_peak_rfd","mean_hd_peak_rfd"],
"Bone Mineral Density": ["Estimated bone mass kg"],
           "Age (y)":["Estimated metabolic age"]
}

corr_path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\correlations_age.xlsx"

result_df = top_correlations_per_target(targets=targets,corr_df=cor_matrix,pval_df=corr_pvalues,method_df=method_matrix,top_n=12,excel_path=corr_path)
