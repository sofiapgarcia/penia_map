import pandas as pd

excel_path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"

df = pd.read_excel(excel_path)

valid_ids = {
    103, 110, 68, 23, 30, 59, 3, 27, 26, 111, 112, 113, 114, 115,
    116, 117, 118, 119, 120, 121, 122, 123, 125, 127, 128,
    131, 126, 124, 21, 36, 82, 32, 19, 61, 49, 132, 133, 33, 51, 37,
    109, 29, 9, 10, 5, 6, 20, 1, 135, 134, 8, 136, 15, 46, 45, 11,
    13, 12, 2, 18, 16, 105, 50, 34
}

# Converter para número, caso estejam como "S103", "103", etc.
subjects = pd.to_numeric(
    df["subject"].astype(str).str.extract(r"(\d+)")[0],
    errors="coerce"
)

missing_from_list = sorted(set(subjects.dropna().astype(int)) - valid_ids)

print("IDs que estão no Excel mas NÃO estão na lista:")
print(missing_from_list)

print(f"\nTotal: {len(missing_from_list)}")

print(len(valid_ids))