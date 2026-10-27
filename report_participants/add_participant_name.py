"""
File for updating participant names in the main Excel file using a names reference file.
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd

# File paths
DATA_FILE = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\report_participants_all.xlsx"
NAMES_FILE = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\demo_name_participants.xlsx"

ID_COL = "ID"
NAME_COL = "Name"
PORTUGUESE_NAME_COL = "Name"
ENGLISH_NAME_COL = "English Name"


# Read Excel files
df = pd.read_excel(DATA_FILE)
df_names = pd.read_excel(NAMES_FILE)

# Check required columns
required_data = [ID_COL]
required_names = [ID_COL, PORTUGUESE_NAME_COL, ENGLISH_NAME_COL]

for col in required_data:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the main Excel.")

for col in required_names:
    if col not in df_names.columns:
        raise ValueError(f"Column '{col}' not found in the names Excel.")

# Normalize IDs for matching
df[ID_COL] = df[ID_COL].astype(str).str.strip()
df_names[ID_COL] = df_names[ID_COL].astype(str).str.strip()

# Prefer the Portuguese name; use the English name if missing
portuguese_name = df_names[PORTUGUESE_NAME_COL].astype("string").str.strip()
english_name = df_names[ENGLISH_NAME_COL].astype("string").str.strip()

df_names[NAME_COL] = portuguese_name.mask(
    portuguese_name.isna() | portuguese_name.eq(""),
    english_name
)

# Ensure there is only one name record per ID
if df_names[ID_COL].duplicated().any():
    duplicated_ids = df_names.loc[
        df_names[ID_COL].duplicated(keep=False), ID_COL
    ].unique().tolist()
    raise ValueError(f"Duplicate IDs in names Excel: {duplicated_ids}")

# Remove an existing Name column before updating it
if NAME_COL in df.columns:
    df = df.drop(columns=NAME_COL)

# Add names without losing or reordering participants
df = df.merge(
    df_names[[ID_COL, NAME_COL]],
    on=ID_COL,
    how="left",
    validate="many_to_one",
    sort=False
)

# Report results
missing_names = df[NAME_COL].isna() | df[NAME_COL].astype("string").str.strip().eq("")

print(f"Participants in main Excel: {len(df)}")
print(f"Participants with a name: {(~missing_names).sum()}")
print(f"Participants without a name: {missing_names.sum()}")

# Save back to the main Excel
df.to_excel(DATA_FILE, index=False)

print(f"\nUpdated Excel saved to: {DATA_FILE}")