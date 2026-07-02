
def emg_transfer_function(df, emg_cols=('EMG_Flexor', 'EMG_Extensor'),
                          n_bits=16, vcc=3.0, g_emg=1000):
    """
    Convert EMG ADC values into millivolts (mV).

    transfer function: https://support.pluxbiosignals.com/wp-content/uploads/2021/10/biosignalsplux-Electromyography-EMG-Datasheet.pdf

    :param df: Input DataFrame containing EMG ADC data
    :param emg_cols: Tuple with column names for EMG channels
    :param n_bits: ADC resolution (number of bits)
    :param vcc: Supply voltage (V)
    :param g_emg: Sensor gain
    :return: DataFrame with additional columns containing EMG signals in mV
    """

    df = df.copy()

    for col in emg_cols:
        # Skip channels that don't exist for this condition
        if col not in df.columns:
            continue

        # Convert to float for numerical stability
        adc = df[col].astype(float)

        # Apply EMG transfer function
        emg_v = ((adc / (2**n_bits) - 0.5) * vcc) / g_emg

        # Convert to millivolts
        df[col] = emg_v * 1000

    return df