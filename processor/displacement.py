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
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
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
        print("Attention: This subject is using an aproximation for the displacement conversion")

        # Convert x0 to the correct physical reference
        print("x0_pega", x0)

        N = df_converted[DISPLACEMENT_COLUMN]

        N0 = _get_N0(N)

        Nmax = 13867

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

    #df_converted = _filter_displacement(df_converted)

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

def _correct_displacement(displacement_mm, xmax_physical_mm):
    """
    Correct displacement values by rescaling them to the physical range.

    Used to correct cases where N0 estimated by _calculate_N0 is not
    exactly the real N0, which can produce negative displacement values
    after the linear conversion (typically when hp_source is "potentiometer").

    :param displacement_mm: Displacement signal already converted to millimeters
    :param xmax_physical_mm: Physical maximum displacement (closed handgrip), in millimeters
    :return: Corrected displacement values, rescaled to [0, xmax_physical_mm]
    """
    x = np.asarray(displacement_mm, dtype=float)

    x_min = np.min(x)
    x_max = np.max(x)

    if np.isclose(x_max, x_min):
        raise ValueError(
            "A série é constante e não pode ser reescalada."
        )

    corrected = (
        (x - x_min)
        / (x_max - x_min)
        * xmax_physical_mm
    )

    corrected = np.clip(
        corrected,
        0.0,
        xmax_physical_mm
    )

    return corrected

def _get_N0(displacement):
    pct = 10
    # Sort the signal to group low, transition, and high samples together
    s = np.sort(np.asarray(displacement))
    n = len(s)

    # Number of samples used to estimate each level
    k = max(1, int(n * pct / 100))

    # Median of the highest k samples -> high level estimate
    N0 = np.median(s[-k:])

    return N0

def _filter_displacement(df, fs=1000, cutoff_hz=20, order=4, column=DISPLACEMENT_COLUMN):
    """
    Apply a zero-phase low-pass Butterworth filter to the displacement signal.

    Uses scipy's filtfilt (forward-backward filtering) to remove phase distortion,
    ensuring the filtered signal remains time-aligned with the original.

    :param df: Input DataFrame containing the displacement column (already converted to mm).
    :param fs: Sampling rate (Hz).
    :param cutoff_hz: Cutoff frequency (Hz) of the low-pass filter. Typical range: 10-20 Hz.
    :param order: Order of the Butterworth filter.
    :param column: Name of the displacement column to filter.
    :return: DataFrame with the filtered displacement values.
    """
    df_filtered = df.copy()

    if column not in df_filtered.columns:
        return df_filtered

    nyquist = fs / 2
    normal_cutoff = cutoff_hz / nyquist

    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    signal = df_filtered[column].to_numpy(dtype=float)
    df_filtered[column] = filtfilt(b, a, signal)

    return df_filtered


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

def _correct_displacement(displacement_mm, xmax_physical_mm):
    """
    Correct displacement values by rescaling them to the physical range.

    Used to correct cases where N0 estimated by _calculate_N0 is not
    exactly the real N0, which can produce negative displacement values
    after the linear conversion (typically when hp_source is "potentiometer").

    :param displacement_mm: Displacement signal already converted to millimeters
    :param xmax_physical_mm: Physical maximum displacement (closed handgrip), in millimeters
    :return: Corrected displacement values, rescaled to [0, xmax_physical_mm]
    """
    x = np.asarray(displacement_mm, dtype=float)

    x_min = np.min(x)
    x_max = np.max(x)

    if np.isclose(x_max, x_min):
        raise ValueError(
            "A série é constante e não pode ser reescalada."
        )

    corrected = (
        (x - x_min)
        / (x_max - x_min)
        * xmax_physical_mm
    )

    corrected = np.clip(
        corrected,
        0.0,
        xmax_physical_mm
    )

    return corrected

def _get_N0(displacement):
    pct = 10
    # Sort the signal to group low, transition, and high samples together
    s = np.sort(np.asarray(displacement))
    n = len(s)

    # Number of samples used to estimate each level
    k = max(1, int(n * pct / 100))

    # Median of the highest k samples -> high level estimate
    N0 = np.median(s[-k:])

    return N0





