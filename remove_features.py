import pandas as pd

# ============================================================
# CONFIGURAÇÃO
# ============================================================

# Caminho do ficheiro Excel de entrada
INPUT_FILE = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"

# Nome do ficheiro Excel de saída
OUTPUT_FILE = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe_filtered_without_mean_variables.xlsx"


# ============================================================
# LER O FICHEIRO
# ============================================================

df = pd.read_excel(INPUT_FILE)

print(f"Ficheiro original: {INPUT_FILE}")
print(f"Número inicial de colunas: {len(df.columns)}")


# ============================================================
# DEFINIR AS COLUNAS A REMOVER
# ============================================================

# Remove:
# 1. Todas as colunas de band power (contêm 'Hz')
# 2. Todas as colunas que têm simultaneamente 'hi' e 'normalized'
#
# Exemplo que REMOVE:
#   rep_max_hi_cocontraction_normalized
#   mean_hi_emg_overlap_area_normalized
#
# Exemplo que NÃO REMOVE:
#   rep_max_hi_onset_diff_ms
#   mean_hi_rms_extensor
#   mean_hd_cocontraction_normalized
"""
columns_to_drop = [
    col
    for col in df.columns
    if (
        "hz" in col.lower()
        or (
            "hi" in col.lower()
            and "normalized" in col.lower()
        )
    )
]
"""
columns_to_drop = [
    col
    for col in df.columns
    if (
        "hz" in col.lower()
        or (
            "hi" in col.lower()
            and "normalized" in col.lower()
        )
        or col.lower().startswith("mean_hi")
        or col.lower().startswith("mean_hd")
    )
]


# ============================================================
# IMPRIMIR AS COLUNAS REMOVIDAS
# ============================================================

print("\n" + "=" * 70)
print(f"COLUNAS REMOVIDAS ({len(columns_to_drop)}):")
print("=" * 70)

for col in columns_to_drop:
    print(col)


# ============================================================
# REMOVER AS COLUNAS
# ============================================================

df = df.drop(columns=columns_to_drop)

print("\n" + "=" * 70)
print(f"Número final de colunas: {len(df.columns)}")


# ============================================================
# GUARDAR O NOVO FICHEIRO
# ============================================================

df.to_excel(OUTPUT_FILE, index=False)

print(f"\nFicheiro guardado com sucesso: {OUTPUT_FILE}")
print("=" * 70)