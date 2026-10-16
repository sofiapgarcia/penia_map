"""
Reads an Excel file, classifies each participant into a MeSH age group using
the "Age (y)" column, and computes means by sex (Gender: 0 = female, 1 = male)
for the selected metrics.

Only edit INPUT_FILE below, then run:  python classify_age_and_means.py
"""

import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# SETTINGS - change the file path here
# ----------------------------------------------------------------------------
INPUT_FILE = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"
SHEET = 0
OUTPUT_FILE = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\means_report_participant.xlsx"

AGE_COL = "Age (y)"
SEX_COL = "Gender"
METRICS = ["Global muscle kg", "hi_values_max", "Bone Mineral Density", "peak_power"]

# MeSH age groups: [19,25), [25,45), [45,65), [65,inf)
BINS = [18, 25, 45, 65, np.inf]
LABELS = [
    "Young Adult (18-24)",
    "Adult (25-44)",
    "Middle Aged (45-64)",
    "Aged (>=65)",
]
OUT_OF_RANGE = "Out of range (<19 or missing age)"
SEX_MAP = {0: "Female", 1: "Male"}
# ----------------------------------------------------------------------------


def find_column(df, name):
    """Find a column ignoring case and leading/trailing spaces."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    key = name.strip().lower()
    if key not in lookup:
        sys.exit(f"Column '{name}' not found.\nAvailable columns: {list(df.columns)}")
    return lookup[key]


def main():
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET)

    age_col = find_column(df, AGE_COL)
    sex_col = find_column(df, SEX_COL)
    metric_cols = [find_column(df, m) for m in METRICS]

    # Force numeric types (invalid text becomes NaN)
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df[sex_col] = pd.to_numeric(df[sex_col], errors="coerce")
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 1) Age group classification
    df["Age group (MeSH)"] = (
        pd.cut(df[age_col], bins=BINS, labels=LABELS, right=False)
        .astype(object)
        .fillna(OUT_OF_RANGE)
    )
    df["Sex"] = df[sex_col].map(SEX_MAP)

    # 2) Means by sex
    means_sex = df.groupby("Sex")[metric_cols].mean()
    means_sex.insert(0, "n", df.groupby("Sex").size())

    # Extra: means by age group and sex
    grp = ["Age group (MeSH)", "Sex"]
    means_group_sex = df.groupby(grp)[metric_cols].mean()
    means_group_sex.insert(0, "n", df.groupby(grp).size())

    counts = df.groupby(grp).size().unstack(fill_value=0)

    # Console output
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\nParticipants per age group and sex:\n", counts)
    print("\nMeans by sex:\n", means_sex.round(3))
    print("\nMeans by age group and sex:\n", means_group_sex.round(3))

    n_out = (df["Age group (MeSH)"] == OUT_OF_RANGE).sum()
    if n_out:
        print(f"\nWarning: {n_out} participant(s) with age <18 or missing are out of range.")

    # Save Excel
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Classified data", index=False)
        counts.to_excel(writer, sheet_name="Counts")
        means_sex.round(3).to_excel(writer, sheet_name="Means by sex")
        means_group_sex.round(3).to_excel(writer, sheet_name="Means group x sex")

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()