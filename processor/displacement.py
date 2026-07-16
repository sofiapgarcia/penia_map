"""
Functions to convert displacement signals.

Available Functions
-------------------
[Public]
convert_displacement(...): Convert displacement signal from bits to millimeters

[Private]
_get_calibration_params(...): Look up handle_x0, N0 and Nmax from the calibration table
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

# external imports
import pandas as pd

# internal imports
from constants import DISPLACEMENT_COLUMN

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #

# Column names in the calibration Excel file
POTENTIOMETER_COLUMN = "posição potenciómetro (mm)"
HANDLE_COLUMN = "posição pega (cm)"
NMAX_COLUMN = "Nmax"
N0_COLUMN = "N0"

path_excel = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\correspondencia_posicoes.xlsx"

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def convert_displacement(df, x0, calibration_file=path_excel, xmax=3, tol=1e-6, mode=3):
    """
    Convert displacement signal from bits to millimeters.

    Uses linear calibration:
        x = a * N + b

    where:
        a = (handle_x0 - xmax) / (Nmax - N0)
        b = - a * N0

    handle_x0, N0 and Nmax are looked up in a calibration Excel table,
    matched by the potentiometer position (x0).

    Two modes are supported for determining N0 and Nmax used in the
    conversion:

    mode 1 (default): N0 and Nmax are taken directly from the calibration
        table (fixed, as originally recorded during calibration).

    mode 2: N0 is recomputed per acquisition as the minimum value of the
        displacement column in df (accounts for N0 drift over time).
        Nmax is then derived by preserving the calibrated range:
            Nmax(t) = N0(t) + (Nmax_calib - N0_calib)
        This assumes the sensor's mechanical/electrical range
        (Nmax_calib - N0_calib) is stable even though the zero-point drifts.

    :param df: Input DataFrame containing displacement column
    :param x0: Initial displacement position used for calibration (potentiometer, mm) —
        used as the lookup key in the calibration table
    :param calibration_file: Path to the Excel calibration table (columns:
        "posição potenciómetro (mm)", "posição pega (cm)", "Nmax", "N0")
    :param xmax: Maximum physical displacement value (default = 3 mm)
    :param tol: Tolerance used when matching x0 against the table's potentiometer column
    :param mode: 1 = use N0/Nmax directly from calibration table (original behavior).
        2 = recompute N0 as min(df signal) per acquisition, and derive Nmax
        by preserving the calibrated range (Nmax_calib - N0_calib).
    :return: DataFrame with converted displacement values
    """

    df_converted = df.copy()

    if DISPLACEMENT_COLUMN not in df_converted.columns:
        return df_converted

    # Look up handle_x0, N0 and Nmax for this x0 from the calibration table
    handle_x0, N0_calib, Nmax_calib = _get_calibration_params(x0, calibration_file, tol)

    # Extract raw sensor signal and ensure numeric type
    N = df_converted[DISPLACEMENT_COLUMN].astype(float)

    if mode == 1:
        N0 = N0_calib
        Nmax = Nmax_calib

    elif mode == 2:
        # N0 recomputed per acquisition (drift-aware); Nmax derived by
        # preserving the calibrated range
        N0 = N.median()
        Nmax = N0 - (N0_calib - Nmax_calib)

    elif mode==3:
        N0 = N.median()
        Nmax = N.min()

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 1 or 2.")

    print("mode", mode)
    print("handle_x0", handle_x0)
    print("N0", N0)
    print("Nmax", Nmax)
    print("x_max", xmax)

    # Compute linear calibration coefficients
    a = (handle_x0 - xmax) / (Nmax - N0)
    b = - a * N0

    L = (handle_x0 - xmax) * 10

    # Apply linear transformation to full signal
    absolute_position = a * N + b

    # Shift so that the initial position (x0) becomes 0 —
    # this represents the actual displacement performed by the subject
    #df_converted[DISPLACEMENT_COLUMN] = handle_x0 - absolute_position
    df_converted[DISPLACEMENT_COLUMN] = absolute_position * 10

    print("max converted / L", round(df_converted[DISPLACEMENT_COLUMN].max() / L,3))

    return df_converted


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _get_calibration_params(x0, calibration_file, tol=1e-6):
    """
    Look up handle_x0, N0 and Nmax from the calibration table, matched
    by the potentiometer position (x0).

    :param x0: Potentiometer position used as the lookup key (mm)
    :param calibration_file: Path to the Excel calibration table
    :param tol: Tolerance used when matching x0 against the table's potentiometer column
    :return: Tuple (handle_x0, N0, Nmax) from the matching row
    :raises ValueError: If no row (or more than one row) matches x0 within tol
    """

    calibration_df = pd.read_excel(calibration_file)

    matches = calibration_df[
        (calibration_df[POTENTIOMETER_COLUMN] - x0).abs() <= tol
    ]

    if matches.empty:
        raise ValueError(
            f"No calibration row found for x0={x0} (tol={tol}) in '{calibration_file}'."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple calibration rows found for x0={x0} (tol={tol}) in '{calibration_file}'."
        )

    row = matches.iloc[0]
    return row[HANDLE_COLUMN], row[N0_COLUMN], row[NMAX_COLUMN]



