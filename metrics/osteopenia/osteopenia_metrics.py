"""
Functions for calculating bone-related variables from Tanita data.

Available Functions
-------------------
[Public]
add_bone_variables(...): Calculate Bone Mineral Density (BMD) and return a
    DataFrame containing subject IDs, estimated bone mass, and BMD.
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #
# Path to the Tanita Excel file
OSTEOPENIA_EXCEL_PATH = Path(r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\tanita.xlsx")

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def add_bone_variables() -> pd.DataFrame:
    """
    Read the Tanita Excel file and calculate estimated bone mass and
    Bone Mineral Density (BMD).

    :return: DataFrame containing the subject ID, estimated bone mass,
             and Bone Mineral Density
    """

    # Read the Tanita data
    df = pd.read_excel(OSTEOPENIA_EXCEL_PATH)

    # Assume the first column contains the subject ID
    id_column = df.columns[0]

    # Calculate Bone Mineral Density (bmd equation obtained with NHANES dataset)
    df["Bone Mineral Density"] = (
        1.32662
        - 0.00519 * df["Height(cm)"]
        + 0.30035 * df["Estimated bone mass kg"]
        - 0.00064 * df["Age (y)"]
        - 0.00776 * df["Gender"]
    )

    # Keep only the ID and bone-related variables
    df = df[
        [
            id_column,
            "Estimated bone mass kg",
            "Bone Mineral Density",
        ]
    ].copy()

    return df