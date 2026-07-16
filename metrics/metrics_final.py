from dynapenia.dynapenia_metrics import add_max_column_from_excel
from sarcopenia.sarcopenia_metrics import add_almi
from powerpenia.powerpenia_metrics import add_normalized_peak_power_columns
from osteopenia.osteopenia_metrics import add_estimated_bone_mass

import pandas as pd

path_tanita = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\tanita.xlsx"
output_path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\final_dataframe.xlsx"

# Read the anthropometric data
df_ant = pd.read_excel(path_tanita)

# Keep only the required columns
id_column = df_ant.columns[0]
df_ant = df_ant[[
    id_column,
    "Body mass(kg)",
    "Height(cm)",
    "Estimated metabolic age",
]]

dyna = add_max_column_from_excel()
sarco = add_almi()
power = add_normalized_peak_power_columns(df_ant)
osteo = add_estimated_bone_mass()

# Merge all DataFrames using the subject ID
merged_df = (
    dyna
    .merge(sarco, on=id_column, how="outer")
    .merge(power, on=id_column, how="outer")
    .merge(osteo, on=id_column, how="outer")
)
# Add the estimated metabolic age
merged_df = merged_df.merge(
    df_ant[[id_column, "Estimated metabolic age"]],
    on=id_column,
    how="left",
)

# Save the merged DataFrame
merged_df.to_excel(output_path, index=False)


