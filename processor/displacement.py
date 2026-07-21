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
# internal imports
from constants import DISPLACEMENT_COLUMN

X_MAX= 29

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def convert_displacement_2(df, x0 , hp_source, low_level, high_level, xmax = X_MAX):
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

        # xo in mm
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

    L = x0 - xmax
    print("Curso total (L): ", L)


    # Apply linear transformation to full signal
    absolute_position = a * N + b

    if hp_source == "potentiometer":
        # N0 estimated from regression can be slightly off, producing
        # negative displacement values; rescale to the physical range
        absolute_position = _correct_displacement(absolute_position, xmax_physical_mm=L)

        absolute_position = pd.Series(
            absolute_position,
            index=df_converted.index,
            name=DISPLACEMENT_COLUMN
        )

        absolute_position.to_excel(
            r"C:\Users\Sofia\Desktop\funcao_professor.xlsx"
        )

    df_converted[DISPLACEMENT_COLUMN] = absolute_position

    return df_converted

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

        N = df_converted[DISPLACEMENT_COLUMN]

        N0 = N.max()

        Nmax = N.min()

    elif hp_source == "handle":

        # xo in mm
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






