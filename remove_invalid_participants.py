import pandas as pd

FIRST = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\features_power_dyna.xlsx"
SECOND = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\demographic_english.xlsx"
OUTPUT = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\demographic_64.xlsx"

# Read Excel files
ids_first = pd.read_excel(FIRST)["subject"]
df_second = pd.read_excel(SECOND)

ids_second = df_second["ID"]

# Check IDs present in the first file but missing from the second
missing_second = ids_first[~ids_first.isin(ids_second)].drop_duplicates()

# Check IDs present in the second file but missing from the first
missing_first = ids_second[~ids_second.isin(ids_first)].drop_duplicates()

print(f"IDs in first file: {ids_first.nunique()}")
print(f"IDs in second file: {ids_second.nunique()}")

if missing_second.empty:
    print("All IDs from the first file exist in the second file.")
else:
    print("IDs present in the first file but missing from the second:")
    print(missing_second.to_list())

if missing_first.empty:
    print("All IDs from the second file exist in the first file.")
else:
    print("IDs present in the second file but missing from the first:")
    print(missing_first.to_list())

# Filter the second file using IDs from the first file
df_result = df_second[df_second["ID"].isin(ids_first)]
df_result.to_excel(OUTPUT, index=False)

print(f"Rows in output file: {len(df_result)}")
print(f"Output saved to: {OUTPUT}")