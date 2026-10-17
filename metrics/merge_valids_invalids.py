import pandas as pd

# File paths
FILE_1 = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\invalidos\report_participants_invalids.xlsx"
FILE_2 = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\report_participants_valids.xlsx"
OUTPUT_FILE = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\report_participants_all.xlsx"

ID_COL = "ID"

# Read Excel files
df1 = pd.read_excel(FILE_1)
df2 = pd.read_excel(FILE_2)

# Check ID column
for name, df in [("Excel 1", df1), ("Excel 2", df2)]:
    if ID_COL not in df.columns:
        raise ValueError(f"Column '{ID_COL}' not found in {name}.")

# Normalize IDs
df1[ID_COL] = df1[ID_COL].astype(str).str.strip()
df2[ID_COL] = df2[ID_COL].astype(str).str.strip()

# Check for duplicate IDs
for name, df in [("Excel 1", df1), ("Excel 2", df2)]:
    if df[ID_COL].duplicated().any():
        duplicates = df.loc[
            df[ID_COL].duplicated(keep=False), ID_COL
        ].unique().tolist()
        raise ValueError(f"Duplicate IDs in {name}: {duplicates}")

# Find common columns, preserving Excel 1 column order
common_columns = [
    col for col in df1.columns
    if col in df2.columns
]

if ID_COL not in common_columns:
    raise ValueError("The ID column is not common to both files.")

# Keep only common columns
df1 = df1[common_columns].copy()
df2 = df2[common_columns].copy()

# Compare participants
ids1 = set(df1[ID_COL])
ids2 = set(df2[ID_COL])

print(f"Participants in Excel 1: {len(df1)}")
print(f"Participants in Excel 2: {len(df2)}")
print(f"Participants in both files: {len(ids1 & ids2)}")
print(f"Only in Excel 1: {len(ids1 - ids2)}")
print(f"Only in Excel 2: {len(ids2 - ids1)}")

print(f"\nCommon columns ({len(common_columns)}):")
print(common_columns)

# Combine rows without losing participants
df_merged = pd.concat([df1, df2], ignore_index=True)

# Verify that no rows were lost
assert len(df_merged) == len(df1) + len(df2)

# Save result
df_merged.to_excel(OUTPUT_FILE, index=False)

print(f"\nTotal rows in final Excel: {len(df_merged)}")
print(f"Unique participants in final Excel: {df_merged[ID_COL].nunique()}")
print(f"Saved to: {OUTPUT_FILE}")