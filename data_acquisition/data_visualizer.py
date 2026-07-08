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


def _center_and_normalize(signal):
    """
    Center a signal by removing its mean and normalize it by its standard
    deviation (z-score), so that it becomes comparable to other signals
    regardless of original scale or offset.

    :param signal: pandas Series or array-like with the signal values
    :return: Z-scored signal (mean 0, std 1)
    """
    return (signal - signal.mean()) / signal.std()


def plot_subject_data(subject_data):
    """
    Plot all acquisitions contained in subject_data.

    HD (which already merges the former HD and HD_EMG acquisitions into a
    single DataFrame) is displayed with one subplot per signal. If both
    gACCy and y columns are present, an additional subplot overlays them
    (centered and normalized) for comparison.
    All other acquisitions are plotted individually.

    :param subject_data: Dictionary returned by load_subject_data() /
        apply_saved_sync_indices() (with HD already merged)
    """

    import matplotlib.pyplot as plt

    non_condition_keys = {"potentiometer_position", "HI_values"}

    # ------------------------------------------------------------------
    # Check availability of HD
    # ------------------------------------------------------------------

    has_hd = (
        "HD" in subject_data
        and isinstance(subject_data["HD"], dict)
        and "data" in subject_data["HD"]
    )

    # ------------------------------------------------------------------
    # Plot HD (+ comparison if possible)
    # ------------------------------------------------------------------

    if has_hd:

        hd = subject_data["HD"]["data"]

        comparison_available = "gACCy" in hd.columns and "y" in hd.columns

        n_subplots = 1 + int(comparison_available)

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
        for column in hd.columns:

            if column == "nSeq":
                continue

            signal = -hd[column] if column == "gACCy" else hd[column]

            axes[idx].plot(signal, label=column, linewidth=0.8)

        axes[idx].set_title("HD")
        axes[idx].legend(loc="upper right", ncol=4, fontsize=8)

        idx += 1

        # ---------------- Comparison ----------------
        if comparison_available:
            gaccy_norm = _center_and_normalize(-hd["gACCy"])
            y_norm = _center_and_normalize(hd["y"])

            axes[idx].plot(
                gaccy_norm,
                label="HD gACCy (z-scored)",
                linewidth=1.2
            )

            axes[idx].plot(
                y_norm,
                label="HD_EMG y (z-scored)",
                linewidth=1.2
            )

            axes[idx].set_title("HD gACCy vs HD_EMG y (z-scored)")
            axes[idx].legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Plot remaining acquisitions
    # ------------------------------------------------------------------

    for condition, content in subject_data.items():

        if condition in non_condition_keys:
            continue

        if condition == "HD":
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

