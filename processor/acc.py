"""
Functions to convert and process acc signal.

Available Functions
-----------------

[Public]
process_acc(...): Processes accelerometer signals and removes gravitational components.

[Private]
_process_acc_signal(...): Processes a single accelerometer signal.
_transfer_acc(...): Converts raw accelerometer values into physical units.
_getAccCalibration(...): Retrieves accelerometer calibration values.
_gravitational_filter(...): Estimates the gravitational component of the signal.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from scipy.signal import butter, filtfilt
import numpy as np

# internal imports
from constants import ACC_COLUMNS_HUB , ACC_COLUMNS_HD

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def process_acc(df, fs):
    """
    Process accelerometer signals depending on acquisition type.

    HUB: converts x, y, z channels
    HD: converts gACCx, gACCy, gACCz channels

    Removes gravitational component and applies processing pipeline.

    :param df: Input DataFrame containing accelerometer signals
    :param fs: Sampling frequency in Hz
    :return: DataFrame with processed accelerometer signals
    """

    df_processed = df.copy()

    # -------------------------
    # HUB ACCELEROMETER
    # -------------------------
    hub_columns = [
        column for column in ACC_COLUMNS_HUB
        if column in df.columns
    ]

    for column in hub_columns:
        df_processed[column] = _process_acc_hub(
            df_processed[column].values,
            fs,
            axis=column[-1]
        )

    # -------------------------
    # HD ACCELEROMETER
    # -------------------------
    hd_columns = [
        column for column in ACC_COLUMNS_HD
        if column in df.columns
    ]

    for column in hd_columns:
        df_processed[column] = _process_acc_hd(
            df_processed[column].values,
            fs
        )

    return df_processed


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _process_acc_hub(a, fs, axis):
    """
    Process raw accelerometer data by converting it to physical units,
    removing the gravitational component, and applying bandpass filtering.

    :param a: Raw accelerometer signal
    :param fs: Sampling frequency in Hz
    :param axis: Accelerometer axis to be converted
    :return: Filtered accelerometer signal
    """
    acc_ms = _transfer_acc(a, axis)
    acc_no_gravity = acc_ms - _gravitational_filter(acc_ms, fs)
    return acc_no_gravity

def _process_acc_hd(a, fs, n=16):
    """
    Convert HD accelerometer raw ADC values to physical units (g).

    Acc(g) = (ADC - 2^n / 2) * (8 / 2^n)

    Signal is assumed to be in range [-4g, 4g].

    After conversion, gravitational component is removed.

    :param a: Raw ADC signal
    :param fs: Sampling frequency in Hz
    :param n: Number of ADC bits (default = 16)
    :return: Processed acceleration signal in g
    """

    adc = np.asarray(a)

    midpoint = (2**n) / 2
    scale = 8 / (2**n)

    acc_g = (adc - midpoint) * scale

    acc_no_gravity = acc_g - _gravitational_filter(acc_g, fs)

    return acc_no_gravity

def _transfer_acc(data, axis):
    """
    This Function transforms the bit values into m.s^-2 units from the Accelerometer sensor.
    :param data: array with bit values from the ACC sensor.
    :param axis: str with the axes we want to transform.
    :return: array with the acceleration values in g-force for the specified axis.
    """
    C_min, C_max = _getAccCalibration(axis)
    d = (2 * (data - C_min) / (C_max - C_min)) - 1
    return d *9.8

def _getAccCalibration(axis:str):
    """
    Function that retrieves the Callibration values for the Accelerometer.
    For this we need to have a callibration file that contains the data when performing the callibration protocol.
    This procedure can be found at: https://support.pluxbiosignals.com/wp-content/uploads/2021/11/Accelerometer_ACC_Datasheet.pdf.
    This function works for tri-axial accelerometers.
    :return: tuple with Callibration values for the three axes.
    """
    calib_file = np.loadtxt(r"C:\Users\Sofia\Desktop\Tese\codigo\Acc_calib_new.txt")
    if axis == 'x':
        C_min = min(calib_file[:, 2])
        C_max = max(calib_file[:, 2])
    if axis == 'y':
        C_min = min(calib_file[:, 3])
        C_max = max(calib_file[:, 3])
    if axis == 'z':
        C_min = min(calib_file[:, 4])
        C_max = max(calib_file[:, 4])

    return C_min, C_max

def _gravitational_filter(signal, fs, cutoff=0.3, order=3):
    """
    Estimate the gravitational component of an accelerometer signal using
    a low-pass Butterworth filter.

    The gravitational acceleration varies slowly over time compared to
    dynamic body movements. This function extracts the low-frequency
    component associated with gravity.

    :param signal: Accelerometer signal
    :param fs: Sampling frequency in Hz
    :param cutoff: Low-pass cutoff frequency in Hz
    :param order: Order of the Butterworth filter
    :return: Estimated gravitational component of the signal
    """

    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist

    # Create low-pass Butterworth filter
    b, a = butter(order, normalized_cutoff, btype='low')

    # Apply zero-phase filtering
    gravity = filtfilt(b, a, signal)

    return gravity




