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

X_MAX= 29

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def convert_displacement(df, x0 , hp_source, low_level, high_level, xmax = X_MAX):
    """
    Convert displacement signal from bits to millimeters.

    Uses linear calibration:
        x = a * N + b

    where:
        a = (xmax - x0) / (Nmax - N0)
        b = x0 - a * N0

    :param df: Input DataFrame containing displacement column
    :param x0: position of the open handgrip
    :param hp_source: source of the measurement of the handgrip
    :param xmax: position of the closed handgrip
    :return: DataFrame with converted displacement values
    """

    df_converted = df.copy()

    if DISPLACEMENT_COLUMN not in df_converted.columns:
        return df_converted

    if hp_source == "potentiometer":

        # Convert x0 to the correct physical reference
        x0 = _convert_x0(x0)

        # Calculate N0 from x0 using regression
        N0 = _calculate_N0(x0)

        Nmax = 13867

    elif hp_source == "handle":

        # N0 of the calibration file
        N0 = high_level

        # Nmax of the calibration file
        Nmax = low_level

    # Extract raw sensor signal and ensure numeric type
    N = df_converted[DISPLACEMENT_COLUMN].astype(float)

    # Compute linear calibration coefficients
    a = (x0 - xmax) / (Nmax - N0)
    b = - a * N0

    L = x0 - xmax
    print("Curso total (L): ", L)


    # Apply linear transformation to full signal
    absolute_position = a * N + b
    df_converted[DISPLACEMENT_COLUMN] = absolute_position

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
    return (0.2146 * x0 + 3.0154) * 10

def _calculate_N0(x0):
    """
    Calculate N0 based on a linear relationship with x0.

    :param x0: Input numeric value
    :return: Computed N0 value using the formula 2570.1 * x0 + 11261
    """
    return 2570.1 * x0 + 11261






