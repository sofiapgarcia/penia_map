import pandas as pd
from conversor_tanita_dxa.ICC import compute_icc_pairs
from conversor_tanita_dxa.bland_altman import plot_bland_altman_pairs


VARIABLE_MAPPING = {
    # DXA variable -> TANITA variable
    "Massa_total": "Body mass(kg)",
    "Massa Corporal": "Body mass(kg)",
    "BMI_DXA": "Body mass index (BMI)",
    "Perc_gordura_total": "Global fat %",
    "Perc_gordura_braco_direito": "Arm fat (right) %",
    "Perc_gordura_braco_esquerdo": "Arm fat (left) %",
    "Perc_gordura_perna_direita": "Leg fat (right) %",
    "Perc_gordura_perna_esquerda": "Leg fat (left) %",
    "Perc_gordura_tronco": "Torso fat %",
    "Magro_total": "Global muscle kg",
    "Massa_magra_braco_direito": "Arm muscle (right) kg",
    "Massa_magra_braco_esquerdo": "Arm muscle (left) kg",
    "Massa_magra_perna_direita": "Leg muscle (right) kg",
    "Massa_magra_perna_esquerda": "Leg muscle (left) kg",
    "Magro_tronco": "Torso muscle kg",
    "BMC_total": "Estimated bone mass kg",
    "Altura": "Altura_testes"
}


filepath = r"C:\Users\Sofia\Desktop\Tese\dados\dxa_tanita.xlsx"

# read the data
df = pd.read_excel(filepath, index_col=0)

# Calculate the ICC values
results_icc = compute_icc_pairs(df, VARIABLE_MAPPING, icc_type="ICC2")
print(results_icc.to_string(index=False))

# Plot the Bland Altman graphs with mapped variables
plot_bland_altman_pairs(df, VARIABLE_MAPPING, show_plot=True)