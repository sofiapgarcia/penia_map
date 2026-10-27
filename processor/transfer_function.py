"""
Functions to apply sensor signal transfer functions and convert raw data into physical units.

Available Functions
-------------------
[Public]
apply_all_transfers(...): Applies all available sensor transfer functions to a raw DataFrame.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

# internal imports
from processor.acc import transfer_acc
from processor.displacement import convert_displacement
from processor.load_cell import convert_force
from processor.emg import emg_transfer_function

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def apply_transfer_functions(df, x0 , hp_source , low_level , high_level):
    """
    Apply all sensor transfer functions to a raw DataFrame.

    Order:
    1. Accelerometers (HUB + HD)
    2. Displacement
    3. EMG
    4. Force

    :param df: Raw input DataFrame
    :param x0: handgrip measurement of the open position
    :param hp_source: handle or potentiometer
    :return: Transformed DataFrame
    """

    df_out = df.copy()

    df_out = transfer_acc(df_out)

    df_out = convert_displacement(df_out, x0=x0 , hp_source = hp_source , low_level = low_level, high_level = high_level)

    df_out = emg_transfer_function(df_out)

    df_out = convert_force(df_out)

    return df_out