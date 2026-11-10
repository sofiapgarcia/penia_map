"""
Script to read an Excel file and add an "Age Group" column
based on the "Age (y)" column.

Classification:
    - "YA" (Young Adults): age up to 35 years (inclusive)
    - "A"  (Adults):        age between 35 (exclusive) and 65 years (inclusive)
    - "E"  (Elderly):       age above 65 years

Available Functions
-------------------
[Public]
process_file(...): Read the Excel file, add the age group column and save the result.

[Private]
_classify_age(...): Return the age group (YA, A or E) for a given age.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd

# File paths
INPUT_FILE = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\report_participants_all.xlsx"

AGE_COL = "Age (y)"

# Age groups
BINS = [18, 25, 45, 65, np.inf]
LABELS = [
    "Young Adult (18-24)",
    "Adult (25-44)",
    "Middle Aged (45-64)",
    "Aged (>=65)",
]

# Read Excel
df = pd.read_excel(INPUT_FILE)

# Check age column
if AGE_COL not in df.columns:
    raise ValueError(f"Column '{AGE_COL}' not found in the Excel file.")

# Convert age to numeric
df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")

# Add age group column
df["Age group (MeSH)"] = pd.cut(
    df[AGE_COL],
    bins=BINS,
    labels=LABELS,
    right=False
).astype(object).fillna("Out of range")

# Save Excel, overwriting the original file
df.to_excel(INPUT_FILE, index=False)

print(f"Age group column added successfully.")
print(f"Participants: {len(df)}")
print(df["Age group (MeSH)"].value_counts(dropna=False))