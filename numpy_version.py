import pandas as pd

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe_filtered_without_mean_variables.xlsx"

df = pd.read_excel(filepath, index_col=0)

age = df["Age (y)"]

n_over_80 = (age > 80).sum()

print(f"Number of participants over 80 years old: {n_over_80}")
