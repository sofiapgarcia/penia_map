"""
Script to classify participants with a high BMI (>= 25) according to their body fat percentage.

Participants from the Tanita file are first filtered to keep only those whose ID also exists in the final dataframe.
Then, among participants with BMI >= 25, excess body fat is defined by a sex-specific cutoff
(Male: >= 25%; Female: >= 35%) and each participant is assigned to one of four classes:
    - Overweight + excess body fat
    - Overweight + normal body fat
    - Obesity + excess body fat
    - Obesity + normal body fat

Counts, percentages and a BMI category x body fat cross-tabulation are printed to the console.

Input:  tanita.xlsx, final_dataframe.xlsx (used only to obtain the valid IDs)
Output: console summary
"""

import numpy as np
import pandas as pd

# -----------------------------
# FILES
# -----------------------------

EXCEL_PATH = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\tanita.xlsx"
path_ids = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"

df = pd.read_excel(EXCEL_PATH)
df_ids = pd.read_excel(path_ids)

# -----------------------------
# FILTER BY IDs
# -----------------------------

# IDs that exist in the other Excel file
valid_ids = df_ids["ID"].dropna().unique()

# Keep only Tanita participants
# that also exist in the other Excel file
df = df[df["ID"].isin(valid_ids)].copy()

print(f"Participants in Tanita: {len(pd.read_excel(EXCEL_PATH))}")
print(f"IDs in the other Excel: {len(valid_ids)}")
print(f"Participants after matching: {len(df)}")

# -----------------------------
# VARIABLES
# -----------------------------

bmi_col = "Body mass index (BMI)"
fat_col = "Global fat %"
gender_col = "Gender"

# Only BMI >= 25
df_bmi_high = df[df[bmi_col] >= 25].copy()

# Gender: 1 = Male, 2 = Female
df_bmi_high["fat_cutoff"] = np.where(
    df_bmi_high[gender_col] == 1,
    25,
    35
)

# Excess body fat
df_bmi_high["excess_body_fat"] = (
    df_bmi_high[fat_col] >= df_bmi_high["fat_cutoff"]
)

# -----------------------------
# CLASSIFICATION
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