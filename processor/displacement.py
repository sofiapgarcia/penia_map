"""
Functions to convert displacement signals.

Available Functions
-------------------
[Public]
convert_displacement(...): Convert displacement signal from bits to millimeters

[Private]
_convert_x0(...): Convert potentiometer measurement to handle measurement
_calculate_N0(...): Calculate N0 using a linear relationship with x0
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

# internal imports
from constants import DISPLACEMENT_COLUMN

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def convert_displacement(df, x0, xmax=1, Nmax=13867):
    """
    Convert displacement signal from bits to millimeters.

    Uses linear calibration:
        x = a * N + b

    where:
        a = (xmax - x0) / (Nmax - N0)
        b = x0 - a * N0

    :param df: Input DataFrame containing displacement column
    :param x0: Initial displacement position used for calibration (potentiometer)
    :param xmax: Maximum physical displacement value (default = 1 mm)
    :param Nmax: Maximum bit value for calibration (default = 13800 bits)
    :return: DataFrame with converted displacement values
    """

    df_converted = df.copy()

    if DISPLACEMENT_COLUMN not in df_converted.columns:
        return df_converted

    # Convert x0 to the correct physical reference
    handle_x0 = _convert_x0(x0)

    # Calculate N0 from x0 using regression
    N0 = _calculate_N0(x0)

    # Extract raw sensor signal and ensure numeric type
    N = df_converted[DISPLACEMENT_COLUMN].astype(float)

    # Compute linear calibration coefficients
    a = (xmax - handle_x0) / (Nmax - N0)
    b = handle_x0 - a * N0

    # Apply linear transformation to full signal
    absolute_position = a * N + b

    # Shift so that the initial position (x0) becomes 0 —
    # this represents the actual displacement performed by the subject
    df_converted[DISPLACEMENT_COLUMN] = handle_x0 - absolute_position

    return df_converted


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _convert_x0(x0):
    """
    Convert x0 from potentiometer measurement to handle measurement.

    :param x0: Input numeric value
    :return: Transformed value computed as 0.2146 * x0 + 3.0154
    """
    return 0.2146 * x0 + 3.0154

def _calculate_N0(x0):
    """
    Calculate N0 based on a linear relationship with x0.

    :param x0: Input numeric value
    :return: Computed N0 value using the formula 2570.1 * x0 + 11261
    """
    return 2570.1 * x0 + 11261



