from scipy.signal import butter, filtfilt
from matplotlib import pyplot as plt
import numpy as np

EMG_CHANNELS = ["EMG_Flexor", "EMG_Extensor"]

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




def process_emg(subject_data, fs, non_condition_keys=None, show_plot=False):
    """
    Process EMG channels (EMG_Flexor, EMG_Extensor) for every condition in
    subject_data: remove the DC offset and apply a 4th-order Butterworth
    bandpass filter (10-499 Hz) to preserve the physiological content of the
    surface EMG signal. Filtered signals are added as new columns
    (e.g. 'EMG_Flexor_filt').

    :param subject_data: Dictionary as returned by load_subject_data
    :param fs: Sampling frequency (Hz) of the EMG signals
    :param non_condition_keys: Set of keys in subject_data to ignore
        (default: {"potentiometer_position", "HI_values"})
    :param show_plot: If True, shows raw vs filtered EMG on the same graph,
        for each channel and condition
    :return: subject_data with filtered EMG columns added
    """

    if non_condition_keys is None:
        non_condition_keys = {"potentiometer_position", "HI_values"}

    for condition, content in subject_data.items():

        if condition in non_condition_keys:
            continue

        df = content["data"]

        for channel in EMG_CHANNELS:

            if channel not in df.columns:
                continue

            filtered = _bandpass_filter_emg(df[channel].values, fs)
            df[f"{channel}_filt"] = filtered

            if show_plot:
                _plot_emg_processing(condition, channel, df[channel].values, filtered)

    return subject_data

def _bandpass_filter_emg(signal, fs, lowcut=10, highcut=499, order=4):
    """
    Remove the DC offset and apply a 4th-order Butterworth bandpass filter
    to a raw EMG signal.

    :param signal: 1D array with the raw EMG signal
    :param fs: Sampling frequency (Hz)
    :param lowcut: Lower cutoff frequency (Hz)
    :param highcut: Upper cutoff frequency (Hz)
    :param order: Filter order
    :return: Filtered signal (same length as input)
    """

    offset_removed = signal - np.mean(signal)

    nyquist = fs / 2
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")

    return filtfilt(b, a, offset_removed)


def _plot_emg_processing(condition, channel, raw, filtered):
    """
    Plot the raw and filtered EMG signal on the same graph for visual
    comparison.

    :param condition: Name of the condition (e.g. "HD", "HI", "MVC")
    :param channel: EMG channel name (e.g. "EMG_Flexor")
    :param raw: Raw EMG signal (1D array)
    :param filtered: Filtered EMG signal (1D array)
    """

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(raw, linewidth=0.6, alpha=0.6, label="Raw")
    ax.plot(filtered, linewidth=0.8, label="Filtered")

    ax.set_title(f"{condition} - {channel}")
    ax.set_xlabel("Samples")
    ax.legend()

    plt.tight_layout()
    plt.show()