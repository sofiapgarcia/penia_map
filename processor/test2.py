import pandas as pd
from feature_reduction import adaptive_correlation_matrix
from conversor_tanita_dxa import build_dxa_predictor , predict_dxa , plot_r2_distribution_kde

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\dxa_tanita.xlsx"

# read the data
df = pd.read_excel(filepath)

corr_matrix = adaptive_correlation_matrix(df)


models = build_dxa_predictor(df)

filepath_without_dxa = r"C:\Users\Sofia\Desktop\Tese\dados\tanita_sem_dxa.xlsx"
tanita_sample = pd.read_excel(filepath)

# remove participant column ID
tanita_sample = tanita_sample.drop(columns=["ID"], errors="ignore")
df_with_dxa = predict_dxa(models, tanita_sample)

df_with_dxa.to_excel(r"C:\Users\Sofia\Desktop\Tese\dxa_predicted.xlsx")

plot_r2_distribution_kde(models)

print("oi")

