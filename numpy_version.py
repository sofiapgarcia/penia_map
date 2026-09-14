import numpy as np
print(np.__version__)
import scipy

print(scipy.__version__)
import pandas as pd


filepath = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe_filtered_without_mean_variables.xlsx"


# 1. Ler o Excel
df = pd.read_excel(filepath)
df = df.drop(columns=["ID"])
print(len(df.columns))