import pandas as pd
from feature_reduction import adaptive_correlation_matrix
from conversor_tanita_dxa import build_dxa_predictor , predict_dxa , plot_r2_distribution_histogram

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\dxa_tanita.xlsx"

# read the data
df = pd.read_excel(filepath, index_col=0)

# build the regression models to convert tanita to dxa variables
models = build_dxa_predictor(df)

# samples of tanita data with no dxa data
filepath_without_dxa = r"C:\Users\Sofia\Desktop\Tese\dados\tanita_sem_dxa.xlsx"
tanita_sample = pd.read_excel(filepath_without_dxa, index_col=0)


# calculate the dxa variables
df_with_dxa = predict_dxa(models, tanita_sample)

# save the prediction od dxa variables
df_with_dxa.to_excel(r"C:\Users\Sofia\Desktop\Tese\dxa_predicted.xlsx")

# plot of r2 values of all models
plot_r2_distribution_histogram(models)


