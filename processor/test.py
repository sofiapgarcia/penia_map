from data_preparation import analyze_excel_file , encode_categorical_columns , plot_outlier_details , identify_outlier_ids
from data_visualization import plot_distribution
import matplotlib.pyplot as plt


#from ml_pipeline import feature_selection_filters

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\dataset.xlsx"

df = analyze_excel_file(filepath)
df_encoded = encode_categorical_columns(df)
#df_features = df_encoded.drop(columns=["ID"], errors="ignore")
lista=['Taxa_A_G', 'Gordo_pernas', 'Perc_gordura_dif_bracos', 'Massa_dif_bracos', 'Massa_gordura_bracos', 'Massa_gordura_braco_direito', 'Massa_gordura_braco_esquerdo', 'Massa_gordura_dif_bracos', 'Massa_gordura_pernas', 'Massa_gordura_perna_direita', 'Massa_gordura_perna_esquerda', 'Massa_gordura_ginoide', 'Massa_magra_dif_bracos', 'BMC_braco_direito', 'BMC_dif_bracos', 'BMC_dif_pernas', 'BMD_bracos', 'BMD_coluna', 'Perc_gordura_regiao_total', 'Massa_tronco_direito', 'Massa_tronco_esquerdo', 'RSMI', 'VAT_volume', 'VAT_massa_gordura', 'TAS_area', 'Massa_magra_dif', 'Hand Grip_F', 'Peak_Power_SQ_F']

#plot_outlier_details(df_encoded,lista)
identify_outlier_ids(df_encoded)
plot_distribution(df_encoded, "Estimated metabolic age")


#print("oi")