"""
Functions to convert displacement signals

Available Functions
-------------------
[Public]
function1(...):

[Private]
_function2(...):
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

# internal imports
from constants import DISPLACEMENT_COLUMN

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def convert_displacement(df, x0, xmax=1, Nmax=13800):
    """
    Convert displacement signal from bits to millimeters.

    Uses linear calibration:
        x = a * N + b

    where:
        a = (xmax - x0) / (Nmax - N0)
        b = x0 - a * N0

    :param df: Input DataFrame containing displacement column
    :param x0: Initial displacement position used for calibration
    :param xmax: Maximum physical displacement value (default = 1 mm)
    :param Nmax: Maximum bit value for calibration (default = 13800 bits)
    :return: DataFrame with converted displacement values
    """

    df_converted = df.copy()

    if DISPLACEMENT_COLUMN not in df_converted.columns:
        return df_converted

    # Convert x0 to the correct physical reference
    x0 = _convert_x0(x0)

    # Calculate N0 from x0 using regression
    N0 = _calculate_N0(x0)

    # Extract raw sensor signal and ensure numeric type
    N = df_converted[DISPLACEMENT_COLUMN].astype(float)

    # Compute linear calibration coefficients
    a = (xmax - x0) / (Nmax - N0)
    b = x0 - a * N0

    # Apply linear transformation to full signal
    df_converted[DISPLACEMENT_COLUMN] = a * N + b

    return df_converted


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _convert_x0(x0):
    # regressão/conversão do x0 medido
    return ...


def _calculate_N0(x0):
    # regressão que transforma posição em bits
    return ...



