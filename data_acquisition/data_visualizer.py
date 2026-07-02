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


def plot_synchronized_hd_signals(params_hd,
                                 df_hd,
                                 params_hd_emg,
                                 df_hd_emg,
                                 hd_accelerometer=("gACCx", "gACCy", "gACCz"),
                                 hd_emg_accelerometer=("x", "y", "z"),
                                 title=None):
    """
    Plot the synchronized HD and HD_EMG acquisitions.

    The three accelerometer axes are overlaid to visually inspect the
    synchronization quality, while the remaining channels are displayed
    individually.

    :param params_hd: Acquisition parameters of the HD recording
    :param df_hd: Synchronized HD DataFrame
    :param params_hd_emg: Acquisition parameters of the HD_EMG recording
    :param df_hd_emg: Synchronized HD_EMG DataFrame
    :param hd_accelerometer: Tuple containing the HD accelerometer
                             column names (x, y, z)
    :param hd_emg_accelerometer: Tuple containing the HD_EMG
                                 accelerometer column names (x, y, z)
    :param title: Optional figure title
    :return: Matplotlib figure object
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # Sampling rate (both acquisitions should already be synchronized)
    fs = params_hd["sampling rate"]

    # Time vector
    time = np.arange(len(df_hd)) / fs

    # Accelerometer column names
    hd_x, hd_y, hd_z = hd_accelerometer
    emg_x, emg_y, emg_z = hd_emg_accelerometer

    # Create figure
    fig, axes = plt.subplots(
        7,
        1,
        figsize=(15, 18),
        sharex=True
    )

    # ----------------------------------------------------------------------------------
    # Accelerometer X
    # ----------------------------------------------------------------------------------

    if hd_x in df_hd.columns and emg_x in df_hd_emg.columns:

        axes[0].plot(
            time,
            df_hd[hd_x],
            label="HD",
            linewidth=1.5
        )

        axes[0].plot(
            time,
            df_hd_emg[emg_x],
            label="HD_EMG",
            linewidth=1.5,
            alpha=0.8
        )

        axes[0].set_ylabel("Accel X")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # ----------------------------------------------------------------------------------
    # Accelerometer Y
    # ----------------------------------------------------------------------------------

    if hd_y in df_hd.columns and emg_y in df_hd_emg.columns:

        axes[1].plot(
            time,
            df_hd[hd_y],
            linewidth=1.5
        )

        axes[1].plot(
            time,
            df_hd_emg[emg_y],
            linewidth=1.5,
            alpha=0.8
        )

        axes[1].set_ylabel("Accel Y")
        axes[1].grid(True, alpha=0.3)

    # ----------------------------------------------------------------------------------
    # Accelerometer Z
    # ----------------------------------------------------------------------------------

    if hd_z in df_hd.columns and emg_z in df_hd_emg.columns:

        axes[2].plot(
            time,
            df_hd[hd_z],
            linewidth=1.5
        )

        axes[2].plot(
            time,
            df_hd_emg[emg_z],
            linewidth=1.5,
            alpha=0.8
        )

        axes[2].set_ylabel("Accel Z")
        axes[2].grid(True, alpha=0.3)

    # ----------------------------------------------------------------------------------
    # Force
    # ----------------------------------------------------------------------------------

    if "FORCE" in df_hd.columns:

        axes[3].plot(
            time,
            df_hd["FORCE"],
            linewidth=1.5
        )

        axes[3].set_ylabel("Force")
        axes[3].grid(True, alpha=0.3)

    # ----------------------------------------------------------------------------------
    # Displacement
    # ----------------------------------------------------------------------------------

    if "DISPLACEMENT" in df_hd.columns:

        axes[4].plot(
            time,
            df_hd["DISPLACEMENT"],
            linewidth=1.5
        )

        axes[4].set_ylabel("Displacement")
        axes[4].grid(True, alpha=0.3)

    # ----------------------------------------------------------------------------------
    # EMG Extensor
    # ----------------------------------------------------------------------------------

    if "EMG_Extensor" in df_hd_emg.columns:

        axes[5].plot(
            time,
            df_hd_emg["EMG_Extensor"],
            linewidth=1.2
        )

        axes[5].set_ylabel("EMG Ext.")
        axes[5].grid(True, alpha=0.3)

    # ----------------------------------------------------------------------------------
    # EMG Flexor
    # ----------------------------------------------------------------------------------

    if "EMG_Flexor" in df_hd_emg.columns:

        axes[6].plot(
            time,
            df_hd_emg["EMG_Flexor"],
            linewidth=1.2
        )

        axes[6].set_ylabel("EMG Flex.")
        axes[6].set_xlabel("Time (s)")
        axes[6].grid(True, alpha=0.3)

    # Add global title
    if title is not None:

        fig.suptitle(
            title,
            fontsize=14,
            fontweight="bold"
        )

    # Adjust layout
    plt.tight_layout()

    plt.show()

    return fig