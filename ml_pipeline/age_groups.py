"""
Script to read an Excel file and add an "Age Group" column
based on the "Age (y)" column.

Classification:
    - "YA" (Young Adults): age up to 35 years (inclusive)
    - "A"  (Adults):        age between 35 (exclusive) and 65 years (inclusive)
    - "E"  (Elderly):       age above 65 years

Usage:
    python classify_age_group.py "path/to/file.xlsx"

    (Or edit the FILE_PATH variable below and run without arguments.)
"""

import sys
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION — change your file path here if you prefer
# ---------------------------------------------------------------------------
FILE_PATH = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"
AGE_COLUMN_NAME = "Estimated metabolic age"
GROUP_COLUMN_NAME = "Metabolic Age Group"
SHEET_NAME = 0  # sheet index or name to read; 0 = first sheet


def classify_age(age):
    """Return the age group (YA, A or E) for a given age."""
    if pd.isna(age):
        return None
    if age <= 35:
        return "YA"
    elif age <= 65:
        return "A"
    else:
        return "E"


def process_file(input_path):
    # Read the Excel file
    df = pd.read_excel(input_path, sheet_name=SHEET_NAME)

    if AGE_COLUMN_NAME not in df.columns:
        raise ValueError(
            f"Column '{AGE_COLUMN_NAME}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Create the new column
    df[GROUP_COLUMN_NAME] = df[AGE_COLUMN_NAME].apply(classify_age)

    # Define output path (by default, appends "_classified" to the original name)
    output_path = input_path[:-5] + "_age_groups_metabolic.xlsx"

    df.to_excel(output_path, index=False)
    print(f"File successfully saved to: {output_path}")

    # Print a summary of how many rows fall into each age group
    print("\nCounts per age group:")
    counts = df[GROUP_COLUMN_NAME].value_counts(dropna=False)
    for group in ["YA", "A", "E"]:
        print(f"  {group}: {counts.get(group, 0)}")

    return output_path


if __name__ == "__main__":
    process_file(FILE_PATH)