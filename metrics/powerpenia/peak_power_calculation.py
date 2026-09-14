from constants import DISPLACEMENT_COLUMN, FORCE_COLUMN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def process_hd_power(subject_data, show_plot=True, velocity_window_ms=5):
    """
    Process the HD condition: compute velocity (derivative of displacement,
    converted from cm to m/s), smooth it with a zero-phase moving average
    (no time delay introduced), then compute power (smoothed velocity *
    force). Extract the peak power point (with its associated velocity and
    force) and the maximum velocity and maximum force, all restricted to the
    concentric phase (samples whose 'Phase' column starts with 'C').

    Adds 'velocity_raw', 'velocity' (smoothed) and 'power' columns to
    subject_data['HD']['data'], and stores the summary in
    subject_data['power_info'] as a DataFrame.

    :param subject_data: Dictionary as returned by load_subject_data. The HD
        DataFrame must already contain a 'Phase' column (e.g. produced by
        detect_concentric_eccentric_phases), with concentric-phase samples
        labeled 'C1', 'C2', etc.
    :param show_plot: If True, shows the HD force/velocity/power signals with
        the peak power, max velocity and max force points annotated
    :param velocity_window_ms: smoothing window length for the velocity
        moving average, in milliseconds.
    :return: subject_data with 'velocity_raw'/'velocity'/'power' columns
        added to HD, and a new 'power_info' key with the summary DataFrame
    """

    hd_content = subject_data["HD"]
    df = hd_content["data"]
    fs = hd_content["params"]["sampling rate"]

    velocity_raw = _compute_velocity(df[DISPLACEMENT_COLUMN].values, fs)
    df["velocity_raw"] = velocity_raw
    df["velocity"] = _moving_average_centered(velocity_raw, fs, velocity_window_ms)
    df["power"] = df["velocity"] * df[FORCE_COLUMN]

    power_info = _get_power_summary(df)

    if show_plot:
        _plot_hd_power_analysis(df, power_info)

    return power_info


def _compute_velocity(displacement, fs):
    """
    Compute velocity as the derivative of a displacement signal, converting
    the result from mm/s to m/s.

    :param displacement: 1D array with the displacement signal (in mm)
    :param fs: Sampling frequency (Hz)
    :return: Velocity signal in m/s (same length as input)
    """

    dt = 1 / fs
    velocity_mm_s = np.gradient(displacement, dt)

    return velocity_mm_s / 1000


def _moving_average_centered(signal, fs, window_ms):
    """
    Smooths a signal with a centered (zero-phase) moving average, so it does
    not shift the signal in time: each point is replaced by the mean of an
    equal number of samples before and after it, rather than only samples
    before it (which a causal moving average would use, introducing a delay).

    :param signal: 1D array with the signal to smooth.
    :param fs: sampling rate in Hz.
    :param window_ms: smoothing window length, in milliseconds.
    :return: 1D numpy array with the smoothed signal (same length as input).
    """
    window_samples = max(int(round(window_ms / 1000 * fs)), 1)

    # pad the edges by repeating the first/last value, so the average near the
    # start/end isn't computed with fewer points than the rest of the signal
    pad = window_samples // 2
    padded = np.pad(signal, pad, mode='edge')

    kernel = np.ones(window_samples) / window_samples
    smoothed = np.convolve(padded, kernel, mode='same')

    return smoothed[pad:pad + len(signal)]


def _get_power_summary(df):
    """
    Build a summary DataFrame with the peak power point (and its associated
    velocity, force and repetition number), plus the maximum velocity and
    maximum force, restricted to the concentric phase only (rows where
    'Phase' starts with 'C'). Rest ('R') and eccentric ('E...') samples are
    excluded.

    :param df: DataFrame with 'velocity', 'power', 'Phase' and FORCE columns
    :return: DataFrame with one row and columns:
        ['peak_power', 'velocity_at_peak_power', 'force_at_peak_power',
         'rep_at_peak_power', 'max_velocity', 'max_force']
    """

    # restrict to concentric-phase samples only
    is_concentric = df["Phase"].astype(str).str.startswith("C")
    concentric_df = df.loc[is_concentric]

    if concentric_df.empty:
        raise ValueError("No samples with Phase starting with 'C' found to compute the power summary.")

    peak_idx = concentric_df["power"].idxmax()
    max_vel_idx = concentric_df["velocity"].idxmax()
    max_force_idx = concentric_df[FORCE_COLUMN].idxmax()

    summary = {
        "peak_power": df.loc[peak_idx, "power"],
        "velocity_at_peak_power": df.loc[peak_idx, "velocity"],
        "force_at_peak_power": df.loc[peak_idx, FORCE_COLUMN],
        "rep_at_peak_power": int(df.loc[peak_idx, "Phase"].lstrip("C")),
        "max_velocity": df.loc[max_vel_idx, "velocity"],
        "max_force": df.loc[max_force_idx, FORCE_COLUMN],
    }

    return pd.DataFrame([summary])


def _plot_hd_power_analysis_with_velocity_raw(df, power_info):
    """
    Plot the HD force, raw velocity, smoothed velocity and power signals in
    four separate subplots (sharing the x-axis), annotating:
        - Force subplot: force at peak power, and max force (concentric only)
        - Raw velocity subplot: unfiltered velocity signal
        - Smoothed velocity subplot: velocity at peak power and max velocity
          (concentric only), computed on the filtered signal
        - Power subplot: the peak power point (concentric only)

    :param df: DataFrame with 'velocity_raw', 'velocity', 'power', 'Phase'
        and FORCE columns
    :param power_info: Summary DataFrame returned by _get_power_summary
    """

    # same concentric-only restriction as in _get_power_summary, so the
    # plotted points match the values stored in power_info
    is_concentric = df["Phase"].astype(str).str.startswith("C")
    concentric_df = df.loc[is_concentric]
    peak_idx = concentric_df["power"].idxmax()
    max_vel_idx = concentric_df["velocity"].idxmax()
    max_force_idx = concentric_df[FORCE_COLUMN].idxmax()

    info = power_info.iloc[0]

    fig, (ax_force, ax_vel_raw, ax_vel_filt, ax_power) = plt.subplots(
        4, 1, figsize=(12, 11), sharex=True
    )

    # --- Force subplot ---
    ax_force.plot(df[FORCE_COLUMN].values, linewidth=0.8, color="tab:gray")
    ax_force.scatter(peak_idx, info["force_at_peak_power"], color="darkred", zorder=5,
                      label=f"Force @ peak power ({info['force_at_peak_power']:.2f} N)")
    ax_force.scatter(max_force_idx, info["max_force"], color="darkgreen", zorder=5,
                      label=f"Max force ({info['max_force']:.2f} N)")
    ax_force.set_ylabel("Force")
    ax_force.legend(fontsize=8, loc="upper right")

    # --- Raw velocity subplot ---
    ax_vel_raw.plot(df["velocity_raw"].values, linewidth=0.8, color="gray",
                     label="Velocity (raw)")
    ax_vel_raw.set_ylabel("Velocity (m/s)")
    ax_vel_raw.legend(fontsize=8, loc="upper right")

    # --- Filtered velocity subplot ---
    ax_vel_filt.plot(df["velocity"].values, linewidth=1.2, color="tab:blue",
                      label="Velocity (filtered)")
    ax_vel_filt.scatter(peak_idx, info["velocity_at_peak_power"], color="darkred", zorder=5,
                         label=f"Velocity @ peak power ({info['velocity_at_peak_power']:.2f} m/s)")
    ax_vel_filt.scatter(max_vel_idx, info["max_velocity"], color="darkgreen", zorder=5,
                         label=f"Max velocity ({info['max_velocity']:.2f} m/s)")
    ax_vel_filt.set_ylabel("Velocity (m/s)")
    ax_vel_filt.legend(fontsize=8, loc="upper right")

    # --- Power subplot ---
    ax_power.plot(df["power"].values, linewidth=0.8, color="tab:purple")
    ax_power.scatter(peak_idx, info["peak_power"], color="darkred", zorder=5,
                      label=f"Peak power ({info['peak_power']:.2f} W)")
    ax_power.set_ylabel("Power (W)")
    ax_power.set_xlabel("Samples")
    ax_power.legend(fontsize=8, loc="upper right")

    fig.suptitle("HD - Force, Velocity (raw/filtered) and Power (concentric phase)")
    plt.tight_layout()
    plt.show()


def _plot_hd_power_analysis(df, power_info):
    """
    Plot the HD force, raw velocity, smoothed velocity and power signals in
    four separate subplots (sharing the x-axis), annotating:
        - Force subplot: force at peak power, and max force (concentric only)
        - Raw velocity subplot: unfiltered velocity signal
        - Smoothed velocity subplot: velocity at peak power and max velocity
          (concentric only), computed on the filtered signal
        - Power subplot: the peak power point (concentric only)

    :param df: DataFrame with 'velocity_raw', 'velocity', 'power', 'Phase'
        and FORCE columns
    :param power_info: Summary DataFrame returned by _get_power_summary
    """

    # same concentric-only restriction as in _get_power_summary, so the
    # plotted points match the values stored in power_info
    is_concentric = df["Phase"].astype(str).str.startswith("C")
    concentric_df = df.loc[is_concentric]
    peak_idx = concentric_df["power"].idxmax()
    max_vel_idx = concentric_df["velocity"].idxmax()
    max_force_idx = concentric_df[FORCE_COLUMN].idxmax()

    info = power_info.iloc[0]

    fig, (ax_force, ax_vel_filt, ax_power) = plt.subplots(
        3, 1, figsize=(12, 11), sharex=True
    )

    # --- Force subplot ---
    ax_force.plot(df[FORCE_COLUMN].values, linewidth=0.8, color="tab:orange")
    ax_force.scatter(peak_idx, info["force_at_peak_power"], color="darkred", zorder=5,
                      label=f"Force @ peak power ({info['force_at_peak_power']:.2f} N)")
    ax_force.scatter(max_force_idx, info["max_force"], color="darkgreen", zorder=5,
                      label=f"Max force ({info['max_force']:.2f} N)")
    ax_force.set_ylabel("Force (N)")
    ax_force.legend(fontsize=8, loc="upper right")


    # --- Filtered velocity subplot ---
    ax_vel_filt.plot(df["velocity"].values, linewidth=1.2, color="tab:blue")
    ax_vel_filt.scatter(peak_idx, info["velocity_at_peak_power"], color="darkred", zorder=5,
                         label=f"Velocity @ peak power ({info['velocity_at_peak_power']:.2f} m/s)")
    ax_vel_filt.scatter(max_vel_idx, info["max_velocity"], color="darkgreen", zorder=5,
                         label=f"Max velocity ({info['max_velocity']:.2f} m/s)")
    ax_vel_filt.set_ylabel("Velocity (m/s)")
    ax_vel_filt.legend(fontsize=8, loc="upper right")

    # --- Power subplot ---
    ax_power.plot(df["power"].values, linewidth=0.8, color="tab:purple")
    ax_power.scatter(peak_idx, info["peak_power"], color="darkred", zorder=5,
                      label=f"Peak power ({info['peak_power']:.2f} W)")
    ax_power.set_ylabel("Power (W)")
    ax_power.set_xlabel("Samples")
    ax_power.legend(fontsize=8, loc="upper right")

    fig.suptitle("Force, Velocity and Power")
    plt.tight_layout()
    plt.show()