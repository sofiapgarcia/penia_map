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
df_features = df.drop(columns=["ID","Torso muscle kg"], errors="ignore")


cor_matrix , corr_pvalues , method_matrix = adaptive_correlation_matrix(df_features)

targets = {"Estimated metabolic age": ["Age (y)"],
"peak_power": ["velocity_at_peak_power", "force_at_peak_power", "peak_power_mass", "peak_power_height_squared", "max_force", "max_velocity","rep_pp_hd_peak_rfd","mean_hd_peak_rfd"],
"Bone Mineral Density": ["Estimated bone mass kg"],
           "Age (y)":["Estimated metabolic age"]
}

corr_path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\correlations.xlsx"

result_df = top_correlations_per_target(targets=targets,corr_df=cor_matrix,pval_df=corr_pvalues,method_df=method_matrix,top_n=10,excel_path=corr_path)

"""
# get correlation by each dimension of the penia map
#corr_penia = correlation_by_groups(df_features)

df_train , df_test = split_train_test_df(df_features, test_size=0.15 , random_state = 3, target_col="target")

# reduce the number of features based on correlation by group
#df_reduced = select_features_by_group(df_features, selected_groups=["Osteopenia", "Powerpenia", "Dynapenia", "Sarcopenia"] , n_features_per_group=2 , verbose=True)

results = run_for_multiple_targets(
    df_train,
    feature_cols=list(df_train.columns),
    target_cols=["Age (target)", "Estimated metabolic age", "Estimated bone mass kg", "peak_power","peak_power_mass","peak_power_height_squared"],
    criterion="pvalue",
    vif_threshold=5.0,
    exclude_features={
        "peak_power": ["velocity_at_peak_power","force_at_peak_power", "peak_power_mass","peak_power_height_squared","max_velocity","max_force"],
        "peak_power_mass": ["velocity_at_peak_power","force_at_peak_power", "peak_power","max_velocity","max_force"],
        "peak_power_height_squared": ["velocity_at_peak_power","force_at_peak_power", "peak_power","max_velocity","max_force"],
    },
)
results_df = results_to_dataframe(results)
#summarize_results(results)
features_por_target = get_selected_features_dict(results)

#target = "Age (y)"
#target = "Estimated metabolic age"
#target = "Estimated bone mass kg"
target = "peak_power"
features = features_por_target[target] + [target]

df_train = df_train[features]

# apply data augmentation
# Gaussian Copula + SMOGN
#df_train = apply_data_augmentation(df_train, n_samples = 20)


# Model selection
results_summary = run_training_pipeline(df_train, splits_inner=3, splits_outer=5, target_column=target)

"""
#filepath3 = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\results_summary_second_phase.xlsx"

#results_summary.to_excel(filepath3, index=False)

# Evaluation of production model
#evaluate_production_model_simple(df_train, df_test)
#evaluate_production_model_k_fold(df_reduced)


# Additional functions

# plot distribution of one variable
#plot_distribution(df_features, "Estimated metabolic age")
