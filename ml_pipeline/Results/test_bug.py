from ml_pipeline.preprocessing.linear_regression_feature_reduction import (
    run_for_multiple_targets,
    export_results_to_excel,
)

import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import timedelta
from pathlib import Path


inicio = time.time()

# ============================================================
# 1. Ler o Excel
# ============================================================

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe_filtered.xlsx"

df = pd.read_excel(filepath)
df = df.drop(columns=["ID"])

# ============================================================
# 2. Definir targets e features
# ============================================================

target_cols = [
    "Age (y)",
    "Estimated metabolic age",
    "Bone Mineral Density",
    "peak_power",
]

feature_cols = [c for c in df.columns if c not in target_cols]

# ============================================================
# 3. Valores de k_max a experimentar
# ============================================================

k_max_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# ============================================================
# 4. Pasta de resultados
# ============================================================

results_dir = Path(r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results")
results_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 5. Exclusões
# ============================================================

exclude_features = {
    "Age (y)": ["Estimated metabolic age"],

    "Estimated metabolic age": ["Age (y)"],

    "peak_power": [
        "velocity_at_peak_power",
        "force_at_peak_power",
        "peak_power_mass",
        "peak_power_height_squared",
        "max_force",
        "max_velocity",
        "rep_pp_hd_peak_rfd",
        "mean_hd_peak_rfd",
    ],

    "peak_power_mass": [
        "velocity_at_peak_power",
        "force_at_peak_power",
        "peak_power",
        "max_velocity",
        "max_force",
    ],

    "peak_power_height_squared": [
        "velocity_at_peak_power",
        "force_at_peak_power",
        "peak_power",
        "max_velocity",
        "max_force",
    ],

    "Bone Mineral Density": [
        "Estimated bone mass kg",
        "Height(cm)",
    ],
}

exclude_features_all = [
    "ID",
    "Age (y)",
    "Estimated metabolic age",
]

# ============================================================
# 6. Correr o best-subset para cada k_max
# ============================================================

all_results = {}
metrics_rows = []

for k_max in k_max_list:

    print("\n" + "=" * 80)
    print(f"                         k_max = {k_max}")
    print("=" * 80)

    results = run_for_multiple_targets(
        df,
        feature_cols,
        target_cols,

        criterion="r2_adj",
        k_max=k_max,

        corr_prescreen=0.85,
        max_combinations=999_999_999_999_999_999_999_999_900_000_000,

        exclude_features=exclude_features,
        exclude_features_all=exclude_features_all,

        vif_threshold=5,
        n_candidates_to_check=10000,

        verbose=True,
    )

    all_results[k_max] = results

    # --------------------------------------------------------
    # Guardar R2 e R2 ajustado para cada target
    # --------------------------------------------------------

    for target, info in results.items():

        metrics_rows.append({
            "k_max": k_max,
            "target": target,
            "n_features": len(info["selected_features"]),
            "features": ", ".join(info["selected_features"]),
            "R2": info["r2"],
            "R2_adjusted": info["r2_adj"],
            "max_VIF": (
                max(info["vif"].values())
                if info["vif"]
                else None
            ),
            "max_p_value": (
                max(info["p_values"].values())
                if info["p_values"]
                else None
            ),
            "passes_diagnostics": info["passes_diagnostics"],
        })

    # --------------------------------------------------------
    # Guardar os resultados completos deste k_max num Excel
    # --------------------------------------------------------

    output_excel = results_dir / f"best_subset_kmax_{k_max}.xlsx"

    export_results_to_excel(
        results,
        str(output_excel),
        decimals=4,
    )

    print(f"\nResultados guardados em: {output_excel}")


# ============================================================
# 7. Tabela conjunta com todos os k_max
# ============================================================

metrics_df = pd.DataFrame(metrics_rows)

metrics_excel = results_dir / "comparacao_kmax_R2_R2_adjusted.xlsx"

with pd.ExcelWriter(metrics_excel, engine="openpyxl") as writer:

    # Tabela completa
    metrics_df.round(4).to_excel(
        writer,
        sheet_name="all_k_max",
        index=False,
    )

    # Uma folha por target
    for target in target_cols:

        target_df = metrics_df[
            metrics_df["target"] == target
        ].copy()

        # Nome válido para Excel
        sheet_name = target[:31]

        target_df.round(4).to_excel(
            writer,
            sheet_name=sheet_name,
            index=False,
        )

print("\n" + "=" * 80)
print("Excel comparativo guardado em:")
print(metrics_excel)
print("=" * 80)


# ============================================================
# 8. Gráficos: R2 e R2 ajustado por target
# ============================================================

plots_dir = results_dir / "plots_kmax"
plots_dir.mkdir(parents=True, exist_ok=True)

for target in target_cols:

    target_df = metrics_df[
        metrics_df["target"] == target
    ].sort_values("k_max")

    if target_df.empty:
        continue

    plt.figure(figsize=(8, 5))

    plt.plot(
        target_df["k_max"],
        target_df["R2"],
        marker="o",
        label="R²",
    )

    plt.plot(
        target_df["k_max"],
        target_df["R2_adjusted"],
        marker="o",
        label="R² ajustado",
    )

    plt.xlabel("k_max")
    plt.ylabel("R²")
    plt.title(f"{target} — R² vs R² ajustado")
    plt.xticks(k_max_list)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Nome seguro para o ficheiro
    safe_target = (
        target
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )

    plot_path = plots_dir / f"{safe_target}_R2_vs_kmax.png"

    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Gráfico guardado: {plot_path}")


# ============================================================
# 9. Gráfico conjunto opcional: apenas R² ajustado
# ============================================================

plt.figure(figsize=(9, 6))

for target in target_cols:

    target_df = metrics_df[
        metrics_df["target"] == target
    ].sort_values("k_max")

    if target_df.empty:
        continue

    plt.plot(
        target_df["k_max"],
        target_df["R2_adjusted"],
        marker="o",
        label=target,
    )

plt.xlabel("k_max")
plt.ylabel("R² ajustado")
plt.title("R² ajustado em função de k_max")
plt.xticks(k_max_list)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

combined_plot = plots_dir / "R2_adjusted_all_targets_vs_kmax.png"

plt.savefig(combined_plot, dpi=300)
plt.close()

print(f"Gráfico conjunto guardado: {combined_plot}")


# ============================================================
# 10. Tempo de execução
# ============================================================

fim = time.time()

tempo = fim - inicio

print("\n" + "=" * 80)
print("PROCESSO TERMINADO")
print("=" * 80)
print("Tempo de execução:", timedelta(seconds=tempo))