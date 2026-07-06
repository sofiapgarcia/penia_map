"""
Functions to visualize the data.

Available Functions
-------------------
[Public]
plot_sensors_over_time(...): Plot all sensor channels over acquisition time.

[Private]

"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def plot_sensors_over_time(params, df, title=None):
    """
    Plot all sensor channels over acquisition time.

    :param params: Dictionary containing acquisition parameters (e.g., sampling rate)
    :param df: DataFrame containing sensor signals including nSeq and channels
    :param title: Optional title for the plot
    :return: Matplotlib figure object with all sensor signals plotted over time
    """

    # Extract sampling rate from acquisition parameters
    sampling_rate = params["sampling rate"]

    # Number of samples in the dataset
    n_samples = len(df)

    # Create time vector in seconds based on sampling rate
    time_seconds = np.arange(n_samples) / sampling_rate

    # Select all sensor columns except sequence counter
    sensor_cols = [col for col in df.columns if col != 'nSeq']

    # Number of sensors to plot
    n_sensors = len(sensor_cols)

    # Create subplot structure (one row per sensor)
    fig, axes = plt.subplots(
        n_sensors,
        1,
        figsize=(14, 3 * n_sensors),
        sharex=True
    )

    # Ensure axes is iterable even when only one sensor exists
    if n_sensors == 1:
        axes = [axes]

    # Plot each sensor signal over time
    for idx, (ax, sensor) in enumerate(zip(axes, sensor_cols)):

        ax.plot(
            time_seconds,
            df[sensor],
            linewidth=1.5,
            alpha=0.8,
            color=f"C{idx}"
        )

        # Label each subplot with sensor name
        ax.set_ylabel(sensor)

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

    # Label x-axis only on last subplot
    axes[-1].set_xlabel("Time (s)")

    # Add global title if provided
    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    plt.show()

    return fig


def plot_interpolated_signal(df, mask, column, time_column="nSeq", title=None):
    """
    Plot a signal column highlighting interpolated values in red.

    :param df: Interpolated DataFrame
    :param mask: Boolean mask DataFrame marking interpolated values
    :param column: Signal column to plot
    :param time_column: Column to use as x-axis
    :param title: Optional custom title for the plot. Defaults to "Interpolated values - {column}"
    """

    if column == time_column:
        return

    plt.figure(figsize=(12, 4))

    plt.plot(
        df[time_column],
        df[column],
        color="steelblue",
        label=column,
        linewidth=1
    )

    interpolated_points = mask[column]

    plt.scatter(
        df[time_column][interpolated_points],
        df[column][interpolated_points],
        color="red",
        s=10,
        label="Interpolated",
        zorder=5
    )

    plt.xlabel(time_column)
    plt.ylabel(column)
    plt.title(title if title else f"Interpolated values - {column}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_subject_data(subject_data):
    """
    Plot all acquisitions contained in subject_data.

    HD and HD_EMG acquisitions are displayed together for comparison.
    If both are available and contain the required columns, an additional
    subplot overlays HD gACCy and HD_EMG y.
    All other acquisitions are plotted individually.

    :param subject_data: Dictionary returned by load_subject_data()
    """

    import matplotlib.pyplot as plt

    non_condition_keys = {"potentiometer_position", "HI_values"}

    # ------------------------------------------------------------------
    # Check availability of HD / HD_EMG
    # ------------------------------------------------------------------

    has_hd = (
        "HD" in subject_data
        and isinstance(subject_data["HD"], dict)
        and "data" in subject_data["HD"]
    )

    has_hd_emg = (
        "HD_EMG" in subject_data
        and isinstance(subject_data["HD_EMG"], dict)
        and "data" in subject_data["HD_EMG"]
    )

    # ------------------------------------------------------------------
    # Plot HD and HD_EMG (+ comparison if possible)
    # ------------------------------------------------------------------

    if has_hd or has_hd_emg:

        hd = subject_data["HD"]["data"] if has_hd else None
        hd_emg = subject_data["HD_EMG"]["data"] if has_hd_emg else None

        comparison_available = (
            has_hd
            and has_hd_emg
            and "gACCy" in hd.columns
            and "y" in hd_emg.columns
        )

        n_subplots = (
            int(has_hd)
            + int(has_hd_emg)
            + int(comparison_available)
        )

        fig, axes = plt.subplots(
            n_subplots,
            1,
            figsize=(14, 3.5 * n_subplots),
            sharex=True
        )

        if n_subplots == 1:
            axes = [axes]

        idx = 0

        # ---------------- HD ----------------
        if has_hd:

            for column in hd.columns:

                if column == "nSeq":
                    continue

                signal = -hd[column] if column == "gACCy" else hd[column]

                axes[idx].plot(signal, label=column, linewidth=0.8)

            axes[idx].set_title("HD")
            axes[idx].legend(loc="upper right", ncol=4, fontsize=8)

            idx += 1

        # ---------------- HD_EMG ----------------
        if has_hd_emg:

            for column in hd_emg.columns:

                if column == "nSeq":
                    continue

                axes[idx].plot(hd_emg[column], label=column, linewidth=0.8)

            axes[idx].set_title("HD_EMG")
            axes[idx].legend(loc="upper right", ncol=4, fontsize=8)

            idx += 1

        # ---------------- Comparison ----------------
        if comparison_available:

            axes[idx].plot(
                -hd["gACCy"],
                label="HD gACCy",
                linewidth=1.2
            )

            axes[idx].plot(
                hd_emg["y"],
                label="HD_EMG y",
                linewidth=1.2
            )

            axes[idx].set_title("HD gACCy vs HD_EMG y")
            axes[idx].legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Plot remaining acquisitions
    # ------------------------------------------------------------------

    for condition, content in subject_data.items():

        if condition in non_condition_keys:
            continue

        if condition in {"HD", "HD_EMG"}:
            continue

        if (
            not isinstance(content, dict)
            or "data" not in content
        ):
            continue

        df = content["data"]

        plt.figure(figsize=(14, 4))

        for column in df.columns:

            if column == "nSeq":
                continue

            plt.plot(df[column], label=column, linewidth=0.8)

        plt.title(condition)
        plt.legend(loc="upper right", ncol=4, fontsize=8)
        plt.tight_layout()
        plt.show()
