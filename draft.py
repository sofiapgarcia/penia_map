
import pandas as pd

# File paths
FILE_1 = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\means_report_participant.xlsx"
FILE_2 = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\demo_name_participants.xlsx"
FILE_3 = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\date_acquisitions.xlsx"
FILE_4 = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\tanita.xlsx"
OUTPUT = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\means_report_participant_with_name.xlsx"

# Read Excel files
df1 = pd.read_excel(FILE_1)
df2 = pd.read_excel(FILE_2)
df3 = pd.read_excel(FILE_3)
df4 = pd.read_excel(FILE_4)

# Extract ID and names
df_names = df2[["ID", "Name", "English Name"]].copy()

# Use English Name when Name is empty
df_names["Name"] = df_names["Name"].replace(r"^\s*$", pd.NA, regex=True)
df_names["Name"] = df_names["Name"].fillna(df_names["English Name"])

# Keep only ID and Name
df_names = df_names[["ID", "Name"]].drop_duplicates(subset="ID")

# Remove existing Name column from df1, if present
if "Name" in df1.columns:
    df1 = df1.drop(columns="Name")

# Add Name using ID
df1 = df1.merge(
    df_names,
    on="ID",
    how="left",
    validate="many_to_one"
)

# Extract ID and Date
df_dates = df3[["ID", "Date"]].drop_duplicates(subset="ID")

# Remove existing Date column from df1, if present
if "Date" in df1.columns:
    df1 = df1.drop(columns="Date")

# Add Date using ID
df1 = df1.merge(
    df_dates,
    on="ID",
    how="left",
    validate="many_to_one"
)

# Add columns from Tanita that do not already exist in df1
df4 = df4.loc[:, ~df4.columns.duplicated()]
columns_to_add = ["ID"] + [
    col for col in df4.columns
    if col != "ID" and col not in df1.columns
]

df1 = df1.merge(
    df4[columns_to_add],
    on="ID",
    how="left",
    validate="many_to_one"
)

# Check missing names and dates
missing_names = df1.loc[df1["Name"].isna(), "ID"].unique()
missing_dates = df1.loc[df1["Date"].isna(), "ID"].unique()

print("IDs without a matching name:", missing_names)
print("IDs without a matching date:", missing_dates)

if len(missing_names) == 0 and len(missing_dates) == 0:
    print("All IDs have matching Name and Date information.")
else:
    print("Some IDs are missing Name or Date information.")

# Keep only the date, without the time
df1["Date"] = pd.to_datetime(
    df1["Date"], errors="coerce", dayfirst=True
).dt.strftime("%d/%m/%Y")

# Save the resulting dataframe
df1.to_excel(OUTPUT, index=False)

print(f"File saved to: {OUTPUT}")

