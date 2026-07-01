"""
Functions to convert force signal.

Available Functions
-------------------
[Public]
convert_force(...): Convert FORCE signal values using calibration factor.

[Private]
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

# internal imports
from constants import FORCE_COLUMN

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def convert_force(df):
    """
    Convert FORCE signal values using calibration factor.

    F = 0.0158709 * N

    :param df: Input DataFrame containing FORCE column
    :return: DataFrame with converted FORCE values
    """

    df_converted = df.copy()

    factor = 0.0158709 * 0.4

    if FORCE_COLUMN in df_converted.columns:
        df_converted[FORCE_COLUMN] = df_converted[FORCE_COLUMN] * factor

    return df_converted


