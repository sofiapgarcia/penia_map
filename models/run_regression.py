"""
Script for running exhaustive best subset selection across multiple regression targets
and exporting the results to an Excel file.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import time
from datetime import timedelta
# internal imports
from models.feature_reduction_and_linear_regression import run_for_multiple_targets , export_results_to_excel
# ------------------------------------------------------------------------------------------------------------------- #

start = time.time()

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe_filtered_without_mean_variables.xlsx"


# 1. Load the Excel file
df = pd.read_excel(filepath,index_col=0)
# anthropometric or penia map models
#df = df.drop(columns=["Height(cm)","Body mass index (BMI)","Body mass(kg)","Activity level"])

# 2. Define feature and target columns
target_cols = ["Age (y)","Estimated metabolic age","Bone Mineral Density","peak_power"]
feature_cols = [c for c in df.columns if c not in ["ID"]]

# 3. Run exhaustive best subset selection
results = run_for_multiple_targets(
    df, feature_cols, target_cols,
    criterion="r2_adj",
    k_max=6,
    corr_prescreen=0.85,
    max_combinations=100_000_000,
    exclude_features={
        "Age (y)":["Estimated metabolic age"],
        "Estimated metabolic age":["Age (y)"],
        "peak_power": ["velocity_at_peak_power", "force_at_peak_power", "peak_power_mass", "peak_power_height_squared", "max_force", "max_velocity","rep_pp_hd_peak_rfd","mean_hd_peak_rfd"],
        "peak_power_mass": ["velocity_at_peak_power","force_at_peak_power", "peak_power","max_velocity","max_force"],
        "peak_power_height_squared": ["velocity_at_peak_power","force_at_peak_power", "peak_power","max_velocity","max_force"],
        "Bone Mineral Density":["Estimated bone mass kg","Height(cm)"]
    },
    exclude_features_all=["ID","Age (y)","Estimated metabolic age"],
    vif_threshold=5,
    n_candidates_to_check=1000000,
    verbose=True,
)
export_results_to_excel(results, r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\resultados_k6_ant_bmd.xlsx")

end = time.time()

tempo = end - start
print("Execution time:", timedelta(seconds=tempo))