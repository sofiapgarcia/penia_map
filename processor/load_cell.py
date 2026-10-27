"""
Functions for converting and processing force signals.

Available Functions
-------------------
[Public]
convert_force(...): Converts force signal values using a calibration factor.
process_force(...): Centers the force signal around zero and applies a moving average filter, optionally plotting the raw and smoothed signals.
[Private]
_plot_smoothing_check(...): Plots the raw and smoothed force signals to assess the effect of smoothing.
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
from matplotlib import pyplot as plt

# internal imports
from constants import FORCE_COLUMN

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def convert_force(df):
    """
    Convert FORCE signal values using calibration factor.

    F = 0.0158709 * N

    :param df: Input DataFrame containing FORCE column
    :return: DataFrame with converted FORCE values
    """

    df_converted = df.copy()

    factor = 0.0158709

    if FORCE_COLUMN in df_converted.columns:
        df_converted[FORCE_COLUMN] = df_converted[FORCE_COLUMN] * factor

    return df_converted


def process_force(df, fs, window_ms=20, show_plot=False):
    """
    Centers the force column of a DataFrame on zero, then smooths it with a
    simple moving average, to remove sample-level noise before further processing
    (e.g. phase detection).

    Centering subtracts the signal's mean, so it oscillates around zero instead of
    around whatever offset the sensor happened to record.

    A moving average replaces each point by the mean of the points around it, which
    flattens out small up-and-down noise while keeping the overall shape of the
    movement. This is a centered (zero-phase) moving average, so it does not shift
    the signal in time.

    The default window (50 ms = 50 samples at 1000 Hz) is deliberately short compared
    to a full repetition (up to ~4 s = ~4000 samples at 1000 Hz), so it smooths out
    noise without blurring the real shape of the movement. Increase `window_ms` if the
    signal is still noisy after filtering, or decrease it if real detail is being
    blurred out.

    :param df: DataFrame containing a force column (FORCE_COLUMN).
    :param fs: sampling rate in Hz.
    :param window_ms: smoothing window length, in milliseconds.
    :param show_plot: bool, if True plots the raw and smoothed signal for a quick
        visual check.
    :return: 1D numpy array with the centered and smoothed force signal (same
        length as the input column).
    """
    if FORCE_COLUMN not in df.columns:
        return df

    force = df[FORCE_COLUMN].to_numpy()

    # center the signal on zero first, using its mean as the reference offset
    centered = force - force.mean()

    window_samples = max(int(round(window_ms / 1000 * fs)), 1)

    # pad the edges by repeating the first/last value, so the average near the
    # start/end isn't computed with fewer points than the rest of the signal
    pad = window_samples // 2
    padded = np.pad(centered, pad, mode='edge')

    kernel = np.ones(window_samples) / window_samples
    smoothed = np.convolve(padded, kernel, mode='same')
    smoothed = smoothed[pad:pad + len(centered)]

    if show_plot:
        _plot_smoothing_check(centered, smoothed, window_ms)

    df[FORCE_COLUMN] = smoothed

    return df

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _plot_smoothing_check(raw, smoothed, window_ms):
    """
    Plots the raw signal on top and the smoothed signal below, sharing the same
    time (samples) axis, so it's easy to check whether the smoothing window is
    removing noise without blurring away real movement detail.

    :param raw: 1D array with the original (unsmoothed) signal.
    :param smoothed: 1D array with the smoothed signal, same length as raw.
    :param window_ms: smoothing window length used, in milliseconds (shown in title).
    :return: None
    """
    fig, (ax_raw, ax_smooth) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax_raw.plot(raw, color='gray', linewidth=0.8, label='Raw')
    ax_raw.set_ylabel('Force')
    ax_raw.set_title(f'Force smoothing check (window = {window_ms} ms)')
    ax_raw.legend()

    ax_smooth.plot(smoothed, color='tab:blue', linewidth=1.4, label='Smoothed')
    ax_smooth.set_xlabel('Samples')
    ax_smooth.set_ylabel('Force')
    ax_smooth.legend()

    plt.tight_layout()
    plt.show()