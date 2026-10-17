import pandas as pd

# File paths
FILE_1 = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\report_participants_all.xlsx"
FILE_2 = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\tanita.xlsx"

ID_COL = "ID"

# Read Excel files
df1 = pd.read_excel(FILE_1)
df2 = pd.read_excel(FILE_2)

# Check ID column
for name, df in [("Excel 1", df1), ("Excel 2", df2)]:
    if ID_COL not in df.columns:
        raise ValueError(f"Column '{ID_COL}' not found in {name}.")

# Normalize IDs for matching
df1[ID_COL] = df1[ID_COL].astype(str).str.strip()
df2[ID_COL] = df2[ID_COL].astype(str).str.strip()

# Check for duplicate IDs in Excel 2
if df2[ID_COL].duplicated().any():
    duplicates = df2.loc[
        df2[ID_COL].duplicated(keep=False), ID_COL
    ].unique().tolist()
    raise ValueError(f"Duplicate IDs in Excel 2: {duplicates}")

# Identify missing columns
missing_columns = [
    col for col in df2.columns
    if col not in df1.columns and col != ID_COL
]

print(f"Participants in Excel 1: {len(df1)}")
print(f"Participants in Excel 2: {len(df2)}")
print(f"\nColumns to add: {len(missing_columns)}")
print(missing_columns)

# Match participants and add missing columns
if missing_columns:
    df1 = df1.merge(
        df2[[ID_COL] + missing_columns],
        on=ID_COL,
        how="left",
        validate="one_to_one",
        sort=False
    )

# Verify participants were not lost
assert len(df1) == pd.read_excel(FILE_1).shape[0]

# Save to Excel 1
df1.to_excel(FILE_1, index=False)

print("\nParticipants after update:", len(df1))
print("Missing columns added successfully.")
print(f"Updated file: {FILE_1}")