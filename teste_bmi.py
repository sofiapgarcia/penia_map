import numpy as np
import pandas as pd

# -----------------------------
# FICHEIROS
# -----------------------------

path_tanita = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\tanita.xlsx"
path_ids = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"

df = pd.read_excel(path_tanita)
df_ids = pd.read_excel(path_ids)

# -----------------------------
# FILTRAR PELOS IDs
# -----------------------------

# IDs que existem no outro Excel
valid_ids = df_ids["ID"].dropna().unique()

# Manter apenas participantes do Tanita
# que também existem no outro Excel
df = df[df["ID"].isin(valid_ids)].copy()

print(f"Participantes no Tanita: {len(pd.read_excel(path_tanita))}")
print(f"IDs no outro Excel: {len(valid_ids)}")
print(f"Participantes após cruzamento: {len(df)}")

# -----------------------------
# VARIÁVEIS
# -----------------------------

bmi_col = "Body mass index (BMI)"
fat_col = "Global fat %"
gender_col = "Gender"

# Apenas BMI >= 25
df_bmi_high = df[df[bmi_col] >= 25].copy()

# Gender: 1 = Male, 2 = Female
df_bmi_high["fat_cutoff"] = np.where(
    df_bmi_high[gender_col] == 1,
    25,
    35
)

# Excesso de gordura
df_bmi_high["excess_body_fat"] = (
    df_bmi_high[fat_col] >= df_bmi_high["fat_cutoff"]
)

# -----------------------------
# CLASSIFICAÇÃO
# -----------------------------

df_bmi_high["classification"] = np.select(
    [
        (df_bmi_high[bmi_col] < 30) & df_bmi_high["excess_body_fat"],
        (df_bmi_high[bmi_col] < 30) & ~df_bmi_high["excess_body_fat"],
        (df_bmi_high[bmi_col] >= 30) & df_bmi_high["excess_body_fat"],
        (df_bmi_high[bmi_col] >= 30) & ~df_bmi_high["excess_body_fat"]
    ],
    [
        "Overweight + excess body fat",
        "Overweight + normal body fat",
        "Obesity + excess body fat",
        "Obesity + normal body fat"
    ],
    default="Unknown"
)

print("\n--- Classification ---")
print(df_bmi_high["classification"].value_counts())

print("\n--- Percentages ---")
print(
    df_bmi_high["classification"]
    .value_counts(normalize=True)
    .mul(100)
    .round(1)
)

# -----------------------------
# CROSS-TAB
# -----------------------------

df_bmi_high["BMI_category"] = np.where(
    df_bmi_high[bmi_col] < 30,
    "Overweight",
    "Obesity"
)

df_bmi_high["Body_fat_category"] = np.where(
    df_bmi_high["excess_body_fat"],
    "Excess body fat",
    "Normal body fat"
)

print("\n--- BMI category × Body fat ---")

print(
    pd.crosstab(
        df_bmi_high["BMI_category"],
        df_bmi_high["Body_fat_category"],
        margins=True
    )
)