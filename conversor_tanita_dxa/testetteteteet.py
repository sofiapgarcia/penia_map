from dynapenia.dynapenia_metrics import add_max_column_from_excel
from sarcopenia.sarcopenia_metrics import add_almi
from powerpenia.powerpenia_metrics import add_normalized_peak_power_columns
from osteopenia.osteopenia_metrics import add_estimated_bone_mass

import pandas as pd

path_tanita = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\tanita.xlsx"
output_path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"

# ---------------------------------------------------------------------
# Read the anthropometric data
# ---------------------------------------------------------------------

df_ant = pd.read_excel(path_tanita)

# Keep only the required columns
id_column = df_ant.columns[0]
df_ant = df_ant[[
    id_column,
    "Body mass(kg)",
    "Height(cm)",
    "Estimated metabolic age",
]]

# ---------------------------------------------------------------------
# Compute feature dataframes
# ---------------------------------------------------------------------

sarco = add_almi()
power = add_normalized_peak_power_columns(df_ant)
osteo = add_estimated_bone_mass()

# ---------------------------------------------------------------------
# Debug information
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("INITIAL DATAFRAMES")
print("=" * 80)

for name, df in [
    ("df_ant", df_ant),
    ("sarco", sarco),
    ("power", power),
    ("osteo", osteo),
]:
    print(f"\n{name}")
    print(f"Rows: {len(df)}")
    print(f"ID dtype: {df[id_column].dtype}")
    print(f"Subjects: {sorted(df[id_column].tolist())}")

# ---------------------------------------------------------------------
# Merge 1: sarco + power
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("MERGE 1: sarco + power")
print("=" * 80)

print("Subjects in sarco but not in power:")
print(sorted(set(sarco[id_column]) - set(power[id_column])))

print("\nSubjects in power but not in sarco:")
print(sorted(set(power[id_column]) - set(sarco[id_column])))

merged_df = sarco.merge(power, on=id_column, how="inner")

print(f"\nSubjects after merge: {len(merged_df)}")
print(sorted(merged_df[id_column].tolist()))

# ---------------------------------------------------------------------
# Merge 2: + osteo
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("MERGE 2: + osteo")
print("=" * 80)

print("Subjects in merged_df but not in osteo:")
print(sorted(set(merged_df[id_column]) - set(osteo[id_column])))

print("\nSubjects in osteo but not in merged_df:")
print(sorted(set(osteo[id_column]) - set(merged_df[id_column])))

merged_df = merged_df.merge(
    osteo,
    on=id_column,
    how="inner",
)

print(f"\nSubjects after merge: {len(merged_df)}")
print(sorted(merged_df[id_column].tolist()))

# ---------------------------------------------------------------------
# Merge 3: + metabolic age
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("MERGE 3: + Estimated metabolic age")
print("=" * 80)

print("Subjects in merged_df but not in df_ant:")
print(sorted(set(merged_df[id_column]) - set(df_ant[id_column])))

print("\nSubjects in df_ant but not in merged_df:")
print(sorted(set(df_ant[id_column]) - set(merged_df[id_column])))

merged_df = merged_df.merge(
    df_ant[[id_column, "Estimated metabolic age"]],
    on=id_column,
    how="inner",
)

print(f"\nSubjects after merge: {len(merged_df)}")
print(sorted(merged_df[id_column].tolist()))

# ---------------------------------------------------------------------
# Final information
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL DATAFRAME")
print("=" * 80)

print(f"Rows: {len(merged_df)}")
print(f"Columns: {len(merged_df.columns)}")
print("\nSubjects:")
print(sorted(merged_df[id_column].tolist()))

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

merged_df.to_excel(output_path, index=False)

print(f"\nSaved to:\n{output_path}")