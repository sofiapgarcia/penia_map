"""
Functions to convert displacement signals.

Available Functions
-------------------
[Public]
convert_displacement(...): Convert displacement signal from bits to millimeters

[Private]
_get_N0(...): Estimate N0 (open handgrip level) from the signal
_convert_x0(...): Convert potentiometer measurement to handle measurement
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np

# internal imports
from constants import DISPLACEMENT_COLUMN

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #
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
        print("Attention: This subject is using an aproximation for the displacement conversion")
        print("x0_pega", x0)

        N = df_converted[DISPLACEMENT_COLUMN]

        N0 = _get_N0(N)

        Nmax = N.min()

        x0 = _convert_x0(x0)

    elif hp_source == "handle":

        # x0 in mm
        x0 = x0 * 10

        # N0 of the calibration file
        N0 = high_level

        # Nmax of the calibration file
        Nmax = low_level

    # Extract raw sensor signal and ensure numeric type
    N = df_converted[DISPLACEMENT_COLUMN].astype(float)

    # Compute linear calibration coefficients
    a = (x0 - xmax) / (Nmax - N0)
    b = - a * N0


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

def _get_N0(displacement):
    """
    Estimate N0 (open handgrip level) as the median of the highest 10% of samples.

    :param displacement: Raw displacement signal (in bits)
    :return: Estimated N0 value
    """
    pct = 10
    # Sort the signal to group low, transition, and high samples together
    s = np.sort(np.asarray(displacement))
    n = len(s)

    # Number of samples used to estimate each level
    k = max(1, int(n * pct / 100))

    # Median of the highest k samples -> high level estimate
    N0 = np.median(s[-k:])

    return N0



