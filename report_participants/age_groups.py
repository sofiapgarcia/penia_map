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
import pandas as pd

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #
FILE_PATH = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"
AGE_COLUMN_NAME = "Age (y)"
GROUP_COLUMN_NAME = "Age Group"
SHEET_NAME = 0

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def process_file(input_path):
    """
    Read an Excel file, add an age group column based on the age column
    and save the result to a new Excel file.

    :param input_path: Path to the input Excel file (.xlsx)
    :return: Path to the saved output file
    """
    # Read the Excel file
    df = pd.read_excel(input_path, sheet_name=SHEET_NAME)

    if AGE_COLUMN_NAME not in df.columns:
        raise ValueError(
            f"Column '{AGE_COLUMN_NAME}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Create the new column
    df[GROUP_COLUMN_NAME] = df[AGE_COLUMN_NAME].apply(_classify_age)

    # Define output path
    output_path = input_path[:-5] + "_age_groups.xlsx"

    df.to_excel(output_path, index=False)
    print(f"File successfully saved to: {output_path}")

    # Print a summary of how many rows fall into each age group
    print("\nCounts per age group:")
    counts = df[GROUP_COLUMN_NAME].value_counts(dropna=False)
    for group in ["YA", "A", "E"]:
        print(f"  {group}: {counts.get(group, 0)}")

    return output_path

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _classify_age(age):
    """
    Return the age group (YA, A or E) for a given age.

    :param age: Age in years (may be NaN)
    :return: "YA" if age <= 35, "A" if age <= 65, "E" otherwise, or None if age is missing
    """
    if pd.isna(age):
        return None
    if age <= 35:
        return "YA"
    elif age <= 65:
        return "A"
    else:
        return "E"

if __name__ == "__main__":
    process_file(FILE_PATH)