from ml_pipeline.modeling.test_models import (
    run_training_pipeline,
    split_train_test_df,
    evaluate_production_model_simple,
    evaluate_production_model_k_fold,
)
from constants import best_models_all_variables, best_models_penia_map
import pandas as pd
from ml_pipeline.preprocessing.linear_regression_feature_reduction import run_for_multiple_targets , export_results_to_excel

import pandas as pd
import time
from datetime import timedelta

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe_filtered_without_mean_variables.xlsx"
ALL_VARIABLES = True
K_MAX = 6
suff = "ant" if ALL_VARIABLES else "penia_map"
inicio = time.time()

# 1. Ler o Excel
df = pd.read_excel(filepath, index_col=0).dropna()
# remove participant column ID
df = df.drop(columns=["ID"], errors="ignore")

if not ALL_VARIABLES:
    df = df.drop(columns=["Height(cm)","Body mass index (BMI)","Body mass(kg)","Activity level"])

target_cols = ["Age (y)", "Estimated metabolic age", "Bone Mineral Density", "peak_power"]
#target_cols = ["Bone Mineral Density"]
feature_cols = [c for c in df.columns if c not in ["ID"]]

exclude_features = {
    "Age (y)": ["Estimated metabolic age"],
    "Estimated metabolic age": ["Age (y)"],
    "peak_power": ["velocity_at_peak_power", "force_at_peak_power", "peak_power_mass",
                   "peak_power_height_squared", "max_force", "max_velocity",
                   "rep_pp_hd_peak_rfd", "mean_hd_peak_rfd"],
    "peak_power_mass": ["velocity_at_peak_power", "force_at_peak_power", "peak_power",
                         "max_velocity", "max_force"],
    "peak_power_height_squared": ["velocity_at_peak_power", "force_at_peak_power", "peak_power",
                                   "max_velocity", "max_force"],
    "Bone Mineral Density": ["Estimated bone mass kg", "Height(cm)"]
}
exclude_features_all = ["ID", "Age (y)", "Estimated metabolic age"]

# 2. Dividir treino/teste por target e correr a seleção só no treino
all_results = {}
train_dfs = {}
test_dfs = {}

for target in target_cols:
    df_train, df_test = split_train_test_df(
        df, target_col=target, test_size=0.15, random_state=42
    )

    # guardar para uso posterior (ex: avaliar no teste depois)
    train_dfs[target] = df_train
    test_dfs[target] = df_test

    # correr o best subset só para este target, usando o respetivo df_train
    result_target = run_for_multiple_targets(
        df_train, feature_cols, [target],
        criterion="r2_adj",
        k_max=K_MAX,
        corr_prescreen=0.85,
        max_combinations=100_000_000,
        exclude_features=exclude_features,
        exclude_features_all=exclude_features_all,
        vif_threshold=5,
        n_candidates_to_check=1_000_000,
        verbose=True,
    )

    all_results.update(result_target)

# 3. Exportar todos os resultados juntos
export_results_to_excel(all_results,rf"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\resultados_k{K_MAX}_{suff}.xlsx")

fim = time.time()

tempo = fim - inicio
print("Tempo de execução do best subset regression:", timedelta(seconds=tempo))

MODEL_NAME = "LR"
NORMALIZATION = "none"

# Features obtidas a partir dos resultados do best subset
features_dict = {
    target: all_results[target]["selected_features"]
    for target in target_cols
}
print(features_dict)

# Lista para acumular os resultados de todas as targets
all_results = []

for target in target_cols:

    print(f"\n{'='*60}")
    print(f"A processar target: {target}")
    print(f"{'='*60}")

    features = features_dict[target] + [target]

    # usar a MESMA divisão treino/teste já feita para este target
    df_train = train_dfs[target]
    df_test = test_dfs[target]

    df_train_reduced = df_train[features].dropna()
    df_test_reduced = df_test[features].dropna()

    results_df = evaluate_production_model_simple(
        df_train_reduced,
        df_test_reduced,
        model_name=MODEL_NAME,
        target_column=target,
        normalization=NORMALIZATION,
    )

    # Adicionar coluna a identificar a target correspondente
    results_df.insert(0, "target", target)

    all_results.append(results_df)

# Concatenar todos os resultados num único DataFrame
final_results = pd.concat(all_results, ignore_index=True)

print("\nResultados finais (todas as targets):")
print(final_results)

# Guardar tudo no mesmo Excel
filepath_final = rf"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\results_test_k{K_MAX}_{suff}.xlsx"
final_results.to_excel(filepath_final, index=False)

print(f"\nResultados guardados em: {filepath_final}")