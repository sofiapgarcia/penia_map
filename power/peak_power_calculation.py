from constants import DISPLACEMENT_COLUMN, FORCE_COLUMN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def process_hd_power(subject_data, show_plot=True):
    """
    Process the HD condition: compute velocity (derivative of displacement,
    converted from cm to m/s) and power (velocity * force), then extract the
    peak power point (with its associated velocity and force) and the overall
    maximum velocity and force independently of where they occur.

    Adds 'velocity' and 'power' columns to subject_data['HD']['data'], and
    stores the summary in subject_data['power_info'] as a DataFrame.

    :param subject_data: Dictionary as returned by load_subject_data
    :param show_plot: If True, shows the HD velocity signal with the peak
        power, max velocity and max force points annotated
    :return: subject_data with 'velocity'/'power' columns added to HD, and a
        new 'power_info' key with the summary DataFrame
    """

    hd_content = subject_data["HD"]
    df = hd_content["data"]
    fs = hd_content["params"]["sampling rate"]

    df["velocity"] = _compute_velocity(df[DISPLACEMENT_COLUMN].values, fs)
    df["power"] = df["velocity"] * df[FORCE_COLUMN]

    power_info = _get_power_summary(df)

    subject_data["power_info"] = power_info

    if show_plot:
        _plot_hd_power_analysis(df, power_info)

    return subject_data

def _compute_velocity(displacement_cm, fs):
    """
    Compute velocity as the derivative of a displacement signal, converting
    the result from cm/s to m/s.

    :param displacement_cm: 1D array with the displacement signal (in cm)
    :param fs: Sampling frequency (Hz)
    :return: Velocity signal in m/s (same length as input)
    """

    dt = 1 / fs
    velocity_cm_s = np.gradient(displacement_cm, dt)

    return velocity_cm_s / 100


def _get_power_summary(df):
    """
    Build a summary DataFrame with the peak power point (and its associated
    velocity and force), plus the overall maximum velocity and maximum force,
    regardless of whether they coincide with the peak power sample.

    :param df: DataFrame with 'velocity', 'power' and FORCE columns
    :return: DataFrame with one row and columns:
        ['peak_power', 'velocity_at_peak_power', 'force_at_peak_power',
         'max_velocity', 'max_force']
    """

    peak_idx = df["power"].idxmax()
    max_vel_idx = df["velocity"].idxmax()
    max_force_idx = df[FORCE_COLUMN].idxmax()

    summary = {
        "peak_power": df.loc[peak_idx, "power"],
        "velocity_at_peak_power": df.loc[peak_idx, "velocity"],
        "force_at_peak_power": df.loc[peak_idx, FORCE_COLUMN],
        "max_velocity": df.loc[max_vel_idx, "velocity"],
        "max_force": df.loc[max_force_idx, FORCE_COLUMN],
    }

    return pd.DataFrame([summary])

def _plot_hd_power_analysis(df, power_info):
    """
    Plot the HD force, velocity and power signals in three separate subplots
    (sharing the x-axis), annotating:
        - Force subplot: force at peak power, and overall max force
        - Velocity subplot: velocity at peak power, and overall max velocity
        - Power subplot: the peak power point

    :param df: DataFrame with 'velocity', 'power' and FORCE columns
    :param power_info: Summary DataFrame returned by _get_power_summary
    """

    peak_idx = df["power"].idxmax()
    max_vel_idx = df["velocity"].idxmax()
    max_force_idx = df[FORCE_COLUMN].idxmax()

    info = power_info.iloc[0]

    fig, (ax_force, ax_vel, ax_power) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # --- Force subplot ---
    ax_force.plot(df[FORCE_COLUMN].values, linewidth=0.8, color="tab:gray")
    ax_force.scatter(peak_idx, info["force_at_peak_power"], color="darkred", zorder=5,
                      label=f"Force @ peak power ({info['force_at_peak_power']:.2f})")
    ax_force.scatter(max_force_idx, info["max_force"], color="darkgreen", zorder=5,
                      label=f"Max force ({info['max_force']:.2f})")
    ax_force.set_ylabel("Force")
    ax_force.legend(fontsize=8, loc="upper right")

    # --- Velocity subplot ---
    ax_vel.plot(df["velocity"].values, linewidth=0.8, color="tab:blue")
    ax_vel.scatter(peak_idx, info["velocity_at_peak_power"], color="darkred", zorder=5,
                   label=f"Velocity @ peak power ({info['velocity_at_peak_power']:.2f} m/s)")
    ax_vel.scatter(max_vel_idx, info["max_velocity"], color="darkgreen", zorder=5,
                   label=f"Max velocity ({info['max_velocity']:.2f} m/s)")
    ax_vel.set_ylabel("Velocity (m/s)")
    ax_vel.legend(fontsize=8, loc="upper right")

    # --- Power subplot ---
    ax_power.plot(df["power"].values, linewidth=0.8, color="tab:purple")
    ax_power.scatter(peak_idx, info["peak_power"], color="darkred", zorder=5,
                      label=f"Peak power ({info['peak_power']:.2f} W)")
    ax_power.set_ylabel("Power (W)")
    ax_power.set_xlabel("Samples")
    ax_power.legend(fontsize=8, loc="upper right")

    fig.suptitle("HD - Force, Velocity and Power")
    plt.tight_layout()
    plt.show()