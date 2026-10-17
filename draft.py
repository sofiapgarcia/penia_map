import pandas as pd

# File path
FILE = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\report_participants_all.xlsx"

# ID to search
ID_TO_FIND = "14"

# Read Excel
df = pd.read_excel(FILE)

# Check ID column
if "ID" not in df.columns:
    raise ValueError("Column 'ID' not found in the Excel file.")

# Compare IDs as text, ignoring spaces
ids = df["ID"].astype(str).str.strip()
matches = df[ids == str(ID_TO_FIND).strip()]

if not matches.empty:
    print(f"ID {ID_TO_FIND} found!")
    print(matches.to_string(index=False))
else:
    print(f"ID {ID_TO_FIND} not found.")