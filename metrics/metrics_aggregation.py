from sarcopenia.sarcopenia_metrics import add_almi
from powerpenia.powerpenia_metrics import add_normalized_peak_power_columns
from osteopenia.osteopenia_metrics import add_bone_variables

import pandas as pd

path_tanita = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\tanita.xlsx"
path_power_dyna = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\features_power_dyna.xlsx"
output_path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"

# ---------------------------------------------------------------------
# Read the anthropometric data
# ---------------------------------------------------------------------

df_ant = pd.read_excel(path_tanita)

# Keep only the required columns

id_column = df_ant.columns[0]
df_ant = df_ant[[
    id_column,
    "Gender",
    "Age (y)",
    "Body mass(kg)",
    "Height(cm)",
    "Estimated metabolic age","Body mass index (BMI)","Activity level",
]]

# Recode Gender: male = 1, female = 0
df_ant["Gender"] = df_ant["Gender"].replace({2: 0})

# ---------------------------------------------------------------------
# Compute feature dataframes
# ---------------------------------------------------------------------

sarco = add_almi()
power = add_normalized_peak_power_columns(df_ant, path_power_dyna)
osteo = add_bone_variables()

# ---------------------------------------------------------------------
# Check that every subject in power exists in the remaining dataframes
# ---------------------------------------------------------------------

power_subjects = set(power[id_column])

print("\n" + "=" * 80)
print("CHECKING SUBJECT CONSISTENCY")
print("=" * 80)

for name, df in [
    ("sarco", sarco),
    ("osteo", osteo),
    ("df_ant", df_ant),
]:
    missing = sorted(power_subjects - set(df[id_column]))

    if missing:
        print(f"\nERROR: Subjects present in power but missing from {name}:")
        print(missing)
    else:
        print(f"\n{name}: OK")

# ---------------------------------------------------------------------
# Merge all dataframes using POWER as the reference dataframe
# ---------------------------------------------------------------------

merged_df = (
    power
    .merge(sarco, on=id_column, how="left")
    .merge(osteo, on=id_column, how="left")
    .merge(
        df_ant[[id_column,"Age (y)", "Height(cm)","Gender", "Estimated metabolic age" , "Body mass index (BMI)", "Body mass(kg)","Activity level"]],
        on=id_column,
        how="left",
    )
)

# ---------------------------------------------------------------------
# Check whether any power subjects are missing information
# ---------------------------------------------------------------------

rows_with_missing = merged_df[merged_df.isna().any(axis=1)]

print("\n" + "=" * 80)
print("FINAL CHECK")
print("=" * 80)

print(f"Subjects in power: {len(power)}")
print(f"Subjects in final dataframe: {len(merged_df)}")

if rows_with_missing.empty:
    print("\nAll power subjects have complete information.")
else:
    print("\nWARNING: The following power subjects have missing information:")
    print(rows_with_missing[id_column].tolist())

    print("\nMissing columns per subject:")
    for _, row in rows_with_missing.iterrows():
        missing_cols = row[row.isna()].index.tolist()
        print(f"Subject {row[id_column]}: {missing_cols}")

# ---------------------------------------------------------------------
# Create male and female dataframes
# ---------------------------------------------------------------------

male_df = merged_df[merged_df["Gender"] == 1].drop(columns="Gender")
female_df = merged_df[merged_df["Gender"] == 0].drop(columns="Gender")

# Remove Gender from the complete dataframe
#merged_df = merged_df.drop(columns="Gender")

# ---------------------------------------------------------------------
# Save the dataframes
# ---------------------------------------------------------------------

output_path_male = output_path.replace(".xlsx", "_male.xlsx")
output_path_female = output_path.replace(".xlsx", "_female.xlsx")

merged_df.to_excel(output_path, index=False)
male_df.to_excel(output_path_male, index=False)
female_df.to_excel(output_path_female, index=False)

print(f"\nSaved complete dataframe to:\n{output_path}")
print(f"Subjects: {len(merged_df)}")

print(f"\nSaved male dataframe to:\n{output_path_male}")
print(f"Subjects: {len(male_df)}")

print(f"\nSaved female dataframe to:\n{output_path_female}")
print(f"Subjects: {len(female_df)}")