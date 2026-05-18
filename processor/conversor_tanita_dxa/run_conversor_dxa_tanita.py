import pandas as pd
from conversor_tanita_dxa import build_dxa_predictor , predict_dxa , get_model_report


filepath = r"C:\Users\Sofia\Desktop\Tese\dados\dxa_tanita.xlsx"

# read the data
df = pd.read_excel(filepath, index_col=0)

# build the regression models to convert tanita to dxa variables
results = build_dxa_predictor(df)

# models report
get_model_report(results)

# samples of tanita data with no dxa data
filepath_without_dxa = r"C:\Users\Sofia\Desktop\Tese\dados\tanita_sem_dxa.xlsx"
tanita_sample = pd.read_excel(filepath_without_dxa, index_col=0)


# calculate the dxa variables
df_with_dxa = predict_dxa(results, tanita_sample)

# save the prediction od dxa variables
df_with_dxa.to_excel(r"C:\Users\Sofia\Desktop\Tese\dxa_predicted_corrected.xlsx")





