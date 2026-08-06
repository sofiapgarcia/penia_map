from ml_pipeline.preprocessing.data_augmentation import *
from ml_pipeline.preprocessing.feature_reduction import select_features_by_group , adaptive_correlation_matrix
from ml_pipeline.modeling.test_models import run_training_pipeline , split_train_test_df , evaluate_production_model_simple , evaluate_production_model_k_fold
from constants import RANDOM_SEED
from ml_pipeline.preprocessing.linear_regression_feature_reduction import run_for_multiple_targets , summarize_results
import pandas as pd

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"

# read the data
df = pd.read_excel(filepath)

# remove participant column ID
df_features = df.drop(columns=["ID"], errors="ignore")

resultados = run_for_multiple_targets(
    df_features,
    feature_cols=list(df_features.columns),
    target_cols=["Age (y)", "Estimated metabolic age", "Estimated bone mass kg", "peak_power"],
    criterion="pvalue",
    vif_threshold=5.0,
    vif_target_aware=True,
    max_features_ratio=0.3,
    exclude_features={
        "peak_power": ["velocity_at_peak_power","force_at_peak_power", "peak_power_mass","peak_power_height_squared"]
    },
    verbose=False,
)

summarize_results(resultados)
"""
for y, info in results.items():
    print(y, "->", info["selected_features"])
    print("removidas por colinearidade:", info["dropped_by_vif"])
    print("excluídas manualmente:", info["excluded_manually"])
"""