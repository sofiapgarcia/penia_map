"""
Script to remove unwanted feature columns from the final dataframe and save the result as a new Excel file.

The following columns are removed:
    1. All band power columns (names containing 'Hz')
    2. All columns whose names contain both 'hi' and 'normalized'
    3. All mean variables (names starting with 'mean_hi' or 'mean_hd')

Input:  final_dataframe.xlsx
Output: final_dataframe_filtered.xlsx
"""

import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

# Path of the input Excel file
INPUT_FILE = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"

# Path of the output Excel file
OUTPUT_FILE = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe_filtered.xlsx"


# ============================================================
# READ THE FILE
# ============================================================

df = pd.read_excel(INPUT_FILE)

print(f"Original file: {INPUT_FILE}")
print(f"Initial number of columns: {len(df.columns)}")


# ============================================================
# DEFINE THE COLUMNS TO DROP
# ============================================================

# Removes:
# 1. All band power columns (contain 'Hz')
# 2. All columns containing both 'hi' and 'normalized'
# 3. All mean variables (start with 'mean_hi' or 'mean_hd')
#
# Examples that ARE removed:
#   rep_max_hi_cocontraction_normalized
#   mean_hi_emg_overlap_area_normalized
#   mean_hi_rms_extensor
#   mean_hd_cocontraction_normalized
#
# Examples that are NOT removed:
#   rep_max_hi_onset_diff_ms
columns_to_drop = [
    col
    for col in df.columns
    if (
        "hz" in col.lower()
        or (
            "hi" in col.lower()
            and "normalized" in col.lower()
        )
        or col.lower().startswith("mean_hi")
        or col.lower().startswith("mean_hd")
    )
]


# ============================================================
# PRINT THE REMOVED COLUMNS
# ============================================================

print("\n" + "=" * 70)
print(f"REMOVED COLUMNS ({len(columns_to_drop)}):")
print("=" * 70)

for col in columns_to_drop:
    print(col)


# ============================================================
# DROP THE COLUMNS
# ============================================================

df = df.drop(columns=columns_to_drop)

print("\n" + "=" * 70)
print(f"Final number of columns: {len(df.columns)}")


# ============================================================
# SAVE THE NEW FILE
# ============================================================

df.to_excel(OUTPUT_FILE, index=False)

print(f"\nFile saved successfully: {OUTPUT_FILE}")
print("=" * 70)