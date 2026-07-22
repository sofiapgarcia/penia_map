from scipy.signal import butter, filtfilt, iirnotch
from matplotlib import pyplot as plt
import numpy as np

from constants import EMG_Flexor_filt, EMG_Extensor_filt , EMG_Flexor, EMG_Extensor

EMG_CHANNELS_filt = [EMG_Flexor_filt, EMG_Extensor_filt]
EMG_CHANNELS =[EMG_Flexor, EMG_Extensor]

def emg_transfer_function(df,
                          n_bits=16, vcc=3.0, g_emg=1000):
    """
    Convert EMG ADC values into millivolts (mV).

    transfer function: https://support.pluxbiosignals.com/wp-content/uploads/2021/10/biosignalsplux-Electromyography-EMG-Datasheet.pdf

    :param df: Input DataFrame containing EMG ADC data
    :param n_bits: ADC resolution (number of bits)
    :param vcc: Supply voltage (V)
    :param g_emg: Sensor gain
    :return: DataFrame with additional columns containing EMG signals in mV
    """

    df = df.copy()

    for col in EMG_CHANNELS:
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


def process_emg(df, fs, show_plot=False):
    """
    Process EMG channels (EMG_Flexor, EMG_Extensor) in a DataFrame: apply a notch filter at 50 Hz to remove powerpenia-line
    interference, and apply a 4th-order Butterworth bandpass filter
    (10-499 Hz) to preserve the physiological content of the surface EMG
    signal. Filtered signals are added as new columns (e.g. 'EMG_Flexor_filt').

    :param df: DataFrame containing the EMG channel(s) to process
    :param fs: Sampling frequency (Hz) of the EMG signals
    :param show_plot: If True, shows raw vs filtered EMG on the same graph,
        for each channel present
    :return: df with filtered EMG columns added
    """

    for channel in EMG_CHANNELS:

        if channel not in df.columns:
            continue

        notch_filtered = _notch_filter_emg(df[channel].values, fs)
        filtered = _bandpass_filter_emg(notch_filtered, fs)
        df[f"{channel}_filt"] = filtered

        if show_plot:
            _plot_emg_processing(channel, df[channel].values, filtered)

    return df

def get_emg_envelope(subject_data, show_plot=True):
    """
    Compute the linear envelope of the filtered EMG channels for each
    condition (HD, HI, MVC) in subject_data: full-wave rectify the filtered
    EMG signal (absolute value) and apply a 4th-order Butterworth low-pass
    filter with a cutoff frequency of 4 Hz. Envelope signals are added as
    new columns (e.g. 'EMG_Flexor_envelope') to each condition's DataFrame.

    Expects filtered EMG columns (e.g. 'EMG_Flexor_filt') to already exist
    in each condition's DataFrame, as produced by process_emg().

    :param subject_data: Dict with keys 'potentiometer_position',
        'HI_values', 'HD', 'HI', 'MVC', 'power_info'. 'HD', 'HI', 'MVC' are
        each dicts with 'data' (DataFrame) and 'params' (dict containing
        'fs', the sampling frequency)
    :param show_plot: If True, shows filtered EMG with envelope overlaid on
        top, for each channel present in each condition
    :return: subject_data with envelope columns added to HD, HI, MVC DataFrames
    """

    for condition in ("HD", "HI", "MVC"):

        if condition not in subject_data:
            continue

        df = subject_data[condition]["data"]
        fs = subject_data[condition]["params"]["sampling rate"]

        for channel in EMG_CHANNELS:

            filt_col = f"{channel}_filt"

            if filt_col not in df.columns:
                continue

            rectified = np.abs(df[filt_col].values)
            envelope = _lowpass_filter_envelope(rectified, fs)
            df[f"{channel}_envelope"] = envelope

            if show_plot:
                _plot_emg_envelope(f"{condition} - {channel}", df[filt_col].values, envelope)

    return subject_data


def _bandpass_filter_emg(signal, fs, lowcut=10, highcut=499, order=4):
    """
    Apply a 4th-order Butterworth bandpass filter
    to a raw EMG signal.

    Frequency range: https://doi.org/10.1038/s41597-023-02223-x

    :param signal: 1D array with the raw EMG signal
    :param fs: Sampling frequency (Hz)
    :param lowcut: Lower cutoff frequency (Hz)
    :param highcut: Upper cutoff frequency (Hz)
    :param order: Filter order
    :return: Filtered signal (same length as input)
    """

    nyquist = fs / 2
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")

    return filtfilt(b, a, signal)

def _notch_filter_emg(signal, fs, notch_freq=50, quality_factor=30):
    """
    Apply a notch filter to remove powerpenia-line interference from an EMG
    signal (e.g. 50 Hz mains noise).

    :param signal: 1D array with the EMG signal (already bandpass filtered)
    :param fs: Sampling frequency (Hz)
    :param notch_freq: Frequency to remove (Hz)
    :param quality_factor: Quality factor of the notch filter (higher = narrower notch)
    :return: Filtered signal (same length as input)
    """
    offset_removed = signal - np.mean(signal)

    nyquist = fs / 2
    b, a = iirnotch(notch_freq / nyquist, quality_factor)

    return filtfilt(b, a, offset_removed)

def _plot_emg_processing(channel, raw, filtered):
    """
    Plot the raw and filtered EMG signal on the same graph for visual
    comparison.

    :param channel: EMG channel name (e.g. "EMG_Flexor")
    :param raw: Raw EMG signal (1D array)
    :param filtered: Filtered EMG signal (1D array)
    """

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(raw, linewidth=0.6, alpha=0.6, label="Raw")
    ax.plot(filtered, linewidth=0.8, label="Filtered")

    ax.set_title(channel)
    ax.set_xlabel("Samples")
    ax.legend()

    plt.tight_layout()
    plt.show()

def _lowpass_filter_envelope(signal, fs, cutoff=4, order=4):
    """
    Apply a 4th-order Butterworth low-pass filter to a rectified EMG signal
    to obtain its linear envelope.

    :param signal: 1D array with the rectified EMG signal
    :param fs: Sampling frequency (Hz)
    :param cutoff: Cutoff frequency (Hz)
    :param order: Filter order
    :return: Filtered signal (same length as input)
    """

    nyquist = fs / 2
    b, a = butter(order, cutoff / nyquist, btype="low")

    return filtfilt(b, a, signal)

def _plot_emg_envelope(title, filtered, envelope):
    """
    Plot the filtered EMG signal with its envelope overlaid on top.

    :param title: Title for the plot (e.g. "HD - EMG_Flexor")
    :param filtered: Filtered EMG signal (1D array)
    :param envelope: Envelope signal (1D array)
    """

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(filtered, linewidth=0.6, alpha=0.5, label="Filtered")
    ax.plot(envelope, linewidth=1.2, color="red", label="Envelope")

    ax.set_title(title)
    ax.set_xlabel("Samples")
    ax.legend()

    plt.tight_layout()
    plt.show()