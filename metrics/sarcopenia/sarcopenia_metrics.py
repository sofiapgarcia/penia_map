"""
Functions to extract sarcopenia metrics and calculate the Appendicular Lean Mass Index (ALMI).

Available Functions
-------------------
[Public]
add_almi(...): Read muscle mass data from an Excel file, extract the required metrics,
and calculate ALMI by normalizing appendicular lean mass to height squared.

"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #

SARCOPENIA_METRICS = [
    "Global muscle kg",
    "Arm muscle (right) kg",
    "Arm muscle (left) kg",
    "Leg muscle (right) kg",
    "Leg muscle (left) kg"
]

APPENDICULAR_METRICS = [
    "Arm muscle (right) kg",
    "Arm muscle (left) kg",
    "Leg muscle (right) kg",
    "Leg muscle (left) kg",
]
# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def add_almi(tanita_excel_path) -> pd.DataFrame:
    """
    Read the sarcopenia Excel file, extract the required muscle metrics,
    and compute the Appendicular Lean Mass Index (ALMI).

    ALM (Appendicular Lean Mass) is defined as the sum of the lean
    (muscle) mass from both arms and both legs. ALMI is obtained by
    normalizing ALM to height squared (ALM / height_m^2).
    :param tanita_excel_path: path to bioimpedance data
    :return: DataFrame containing the selected sarcopenia metrics and the
             ALMI column
    """

    # Read the sarcopenia Excel file
    df = pd.read_excel(tanita_excel_path)

    # Assume the first column contains the subject ID
    id_column = df.columns[0]

    # Keep only the ID column and the required sarcopenia metrics
    df = df[[id_column, "Height(cm)"] + SARCOPENIA_METRICS].copy()

    # Convert height from centimeters to meters
    height_m = df["Height(cm)"] / 100

    # Compute ALM = sum of arm and leg muscle mass (appendicular lean mass)
    alm = df[APPENDICULAR_METRICS].sum(axis=1)

    # Compute ALMI = ALM / height²
    df["ALMI"] = alm / (height_m**2)

    # Remove the temporary height column
    df = df.drop(columns=["Height(cm)"])

    return df

