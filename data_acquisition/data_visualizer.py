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

    return fig


def plot_interpolated_signal(df, mask, column, time_column="nSeq"):
    """
    Plot a signal column highlighting interpolated values in red.

    :param df: Interpolated DataFrame
    :param mask: Boolean mask DataFrame marking interpolated values
    :param column: Signal column to plot
    :param time_column: Column to use as x-axis
    """

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
    plt.title(f"Interpolated values - {column}")
    plt.legend()
    plt.tight_layout()
    plt.show()