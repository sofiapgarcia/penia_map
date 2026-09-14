from ml_pipeline.preprocessing.linear_regression_feature_reduction import run_for_multiple_targets , export_results_to_excel

import pandas as pd
import time
from datetime import timedelta

inicio = time.time()


filepath = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\train.xlsx"


# 1. Ler o Excel
df = pd.read_excel(filepath)
#df = df.drop(columns=["ID"])
#df = df.drop(columns=["ID","Height(cm)","Body mass index (BMI)","Body mass(kg)","Activity level"])
df = df.drop(columns=["Height(cm)","Body mass index (BMI)","Body mass(kg)","Activity level"])
# 2. Definir as colunas de features (as 180) e os targets
#target_cols = ["Age (y)","Estimated metabolic age","Bone Mineral Density","peak_power"]   # ajusta aos teus targets
target_cols = ["Estimated metabolic age"]
feature_cols = [c for c in df.columns if c not in target_cols]

# 3. Correr com best subset exaustivo
results = run_for_multiple_targets(
    df, feature_cols, target_cols,
    criterion="r2_adj",          # ou "aic" / "r2_adj"
    k_max=5,
    corr_prescreen=0.85,      # aperta se der erro de max_combinations
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
export_results_to_excel(results, r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\resultados_train_penia_map.xlsx")


fim = time.time()

tempo = fim - inicio
print("Tempo de execução:", timedelta(seconds=tempo))