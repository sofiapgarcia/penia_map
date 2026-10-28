"""
Functions to normalize peak power values using body mass and height.

Available Functions
-------------------
[Public]
add_normalized_peak_power_columns(...): Load peak power data from an Excel file and add peak power normalized
by body mass and height squared, using anthropometric data from the input DataFrame.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def add_normalized_peak_power_columns(df: pd.DataFrame, dyna_power_metrics_excel: str) -> pd.DataFrame:
    """
    Read the peak power Excel file and add normalized peak power columns using
    the body mass and height from the input DataFrame.


    :param df: Input DataFrame containing the subject ID, body mass, and height
    :param dyna_power_metrics_excel: Path to power and dyna metrics
    :return: Peak power DataFrame with the normalized peak power columns added

    """

    peak_power_df = pd.read_excel(dyna_power_metrics_excel)

    # Rename ID columns to a common name
    peak_power_df = peak_power_df.rename(columns={peak_power_df.columns[0]: "ID"})

    # Merge body mass and height
    peak_power_df = peak_power_df.merge(
        df[["ID", "Body mass(kg)", "Height(cm)"]],
        on="ID",
        how="left",
    )

    # Convert height from centimeters to meters
    height_m = peak_power_df["Height(cm)"] / 100

    # Normalize peak power by body mass
    peak_power_df["peak_power_mass"] = (
        peak_power_df["peak_power"] / peak_power_df["Body mass(kg)"]
    )

    # Normalize peak power by height squared (m²)
    peak_power_df["peak_power_height_squared"] = (
        peak_power_df["peak_power"] / (height_m**2)
    )

    # Remove temporary columns used for the calculations
    peak_power_df = peak_power_df.drop(columns=["Body mass(kg)", "Height(cm)"])

    return peak_power_df