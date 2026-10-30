"""
Functions for visualizing sensor data and signal processing results.

Available Functions
-------------------
[Public]
plot_subject_data(...): Plot all acquisitions for a subject, pairing raw and
    filtered signals and comparing synchronized accelerometer signals when
    available.
plot_sensors_over_time(...): Plot all sensor channels over acquisition time.
plot_column_over_time(...): Plot a single signal column over acquisition time.
plot_interpolated_signal(...): Plot a signal and highlight interpolated values.

[Private]
_center_and_normalize(...): Center and normalize a signal using its z-score.
_group_signal_columns(...): Group raw signals with their filtered counterparts.
_plot_group(...): Plot a signal group on a shared axis.
_make_subplot_figure(...): Create a figure with vertically stacked subplots.
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #

# Columns to always exclude from plotting (nSeq is metadata; x, z, gACCx,
# gACCz are accelerometer channels not needed in these plots)
ALWAYS_EXCLUDED_COLUMNS = {"nSeq", "x", "z", "gACCx", "gACCz"}

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def plot_subject_data(subject_data):
    """
    Plot all acquisitions contained in subject_data.

    DHD (which already merges the former DHD and DHD_HUB acquisitions into a
    single DataFrame) is displayed with one subplot per signal, except
    that each raw signal is plotted together with its filtered
    counterpart (columns ending in "_filt") in the same subplot, using
    different colors. If both gACCy and target columns are present, an
    additional subplot overlays them (centered and normalized) for
    comparison, and the individual gACCy/target subplots are skipped since
    they are already shown in that overlay. The accelerometer channels
    x, z, gACCx and gACCz are excluded from all plots. All other
    acquisitions follow the same one-subplot-per-signal rule, with
    raw/filtered pairing applied as well.

    :param subject_data: Dictionary returned by load_subject_data() /
        apply_saved_sync_indices() (with DHD already merged)
    """

    non_condition_keys = {"potentiometer_position", "IHD_values"}

    # ------------------------------------------------------------------
    # Check availability of DHD
    # ------------------------------------------------------------------

    has_dhd = (
        "DHD" in subject_data
        and isinstance(subject_data["DHD"], dict)
        and "data" in subject_data["DHD"]
    )

    # ------------------------------------------------------------------
    # Plot DHD (one subplot per signal, raw+filtered paired + comparison)
    # ------------------------------------------------------------------

    if has_dhd:

        dhd = subject_data["DHD"]["data"]

        comparison_available = "gACCy" in dhd.columns and "y" in dhd.columns

        excluded = set(ALWAYS_EXCLUDED_COLUMNS)
        if comparison_available:
            excluded |= {"gACCy", "y"}

        dhd_columns = [column for column in dhd.columns if column not in excluded]
        groups = _group_signal_columns(dhd_columns)

        n_subplots = len(groups) + int(comparison_available)

        if n_subplots > 0:

            fig, axes = _make_subplot_figure(n_subplots)

            idx = 0

            # ---------------- DHD signals (raw + filtered paired) ----------------
            for group in groups:
                title = f"DHD - {group[0]}"
                _plot_group(axes[idx], dhd, group, title)
                idx += 1

            # ---------------- Comparison (overlaid) ----------------
            if comparison_available:
                gaccy_norm = _center_and_normalize(-dhd["gACCy"])
                y_norm = _center_and_normalize(dhd["y"])

                axes[idx].plot(
                    gaccy_norm,
                    label="DHD gACCy (z-scored)",
                    linewidth=1.2
                )

                axes[idx].plot(
                    y_norm,
                    label="DHD_HUB y (z-scored)",
                    linewidth=1.2
                )

                axes[idx].set_title("DHD gACCy vs DHD_HUB y (z-scored)", fontsize=10, pad=8)
                axes[idx].legend(loc="upper right")

            plt.show()

    # ------------------------------------------------------------------
    # Plot remaining acquisitions (one subplot per signal, raw+filtered paired)
    # ------------------------------------------------------------------

    for condition, content in subject_data.items():

        if condition in non_condition_keys:
            continue

        if condition == "DHD":
            continue

        if (not isinstance(content, dict) or "data" not in content):
            continue

        df = content["data"]

        columns = [column for column in df.columns if column not in ALWAYS_EXCLUDED_COLUMNS]
        groups = _group_signal_columns(columns)

        n_subplots = len(groups)

        if n_subplots == 0:
            continue

        fig, axes = _make_subplot_figure(n_subplots)

        for idx, group in enumerate(groups):
            title = f"{condition} - {group[0]}"
            _plot_group(axes[idx], df, group, title)

        plt.show()

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


def plot_column_over_time(params, df, column, title=None):
    """
    Plot a single DataFrame column over acquisition time.

    :param params: Dictionary containing acquisition parameters (e.g., sampling rate)
    :param df: DataFrame containing the signal
    :param column: Name of the column to plot
    :param title: Optional title for the plot
    :return: Matplotlib figure object
    """

    # Check if column exists
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # Extract sampling rate
    sampling_rate = params["sampling rate"]

    # Create time vector
    time_seconds = np.arange(len(df)) / sampling_rate

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot signal
    ax.plot(
        time_seconds,
        df[column],
        linewidth=1.5,
        color="C0"
    )

    # Labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(column)
    ax.grid(True, alpha=0.3)

    # Title
    if title is None:
        title = column

    ax.set_title(title)

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
    matplotlib.rcParams['toolbar'] = 'toolbar2'
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

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _center_and_normalize(signal):
    """
    Center a signal by removing its mean and normalize it by its standard
    deviation (z-score), so that it becomes comparable to other signals
    regardless of original scale or offset.

    :param signal: pandas Series or array-like with the signal values
    :return: Z-scored signal (mean 0, std 1)
    """
    return (signal - signal.mean()) / signal.std()


def _group_signal_columns(columns):
    """
    Group column names so that raw signals are paired with their
    filtered counterparts (columns ending in "_filt").

    :param columns: List of column names to group
    :return: List of groups, where each group is a list of one or
        two column names to be plotted together
    """
    filt_suffix = "_filt"

    raw_columns = [c for c in columns if not c.endswith(filt_suffix)]
    filt_columns = {c for c in columns if c.endswith(filt_suffix)}

    groups = []
    used_filt = set()

    # Pair each raw column with its filtered counterpart, if any
    for raw_col in raw_columns:
        filt_col = f"{raw_col}{filt_suffix}"
        if filt_col in filt_columns:
            groups.append([raw_col, filt_col])
            used_filt.add(filt_col)
        else:
            groups.append([raw_col])

    # Filtered columns without a matching raw column (edge case)
    for filt_col in filt_columns:
        if filt_col not in used_filt:
            groups.append([filt_col])

    return groups


def _plot_group(ax, df, group, title):
    """
    Plot one or two related columns (raw + filtered) on a single axis.

    :param ax: Matplotlib axis to plot on
    :param df: DataFrame containing the columns
    :param group: List with one or two column names to plot
    :param title: Title to set on the axis
    """
    colors = ["tab:blue", "tab:red"]

    for color, column in zip(colors, group):
        signal = -df[column] if column == "gACCy" else df[column]
        ax.plot(signal, label=column, linewidth=0.8, color=color)

    ax.set_title(title, fontsize=10, pad=8)
    ax.legend(loc="upper right", fontsize=8)


def _make_subplot_figure(n_subplots, height_per_subplot=2.2, max_fig_height=30):
    """
    Create a figure with the given number of vertically stacked subplots,
    sized so titles and axes have enough room to avoid overlap.

    :param n_subplots: Number of subplots to create
    :param height_per_subplot: Figure height allocated per subplot (inches)
    :param max_fig_height: Maximum total figure height (inches), to avoid
        unreasonably tall figures when there are many subplots
    :return: Tuple of (fig, axes) where axes is always a list
    """
    fig_height = min(height_per_subplot * n_subplots, max_fig_height)

    fig, axes = plt.subplots(
        n_subplots,
        1,
        figsize=(14, fig_height),
        sharex=True,
        constrained_layout=True
    )

    if n_subplots == 1:
        axes = [axes]

    fig.get_layout_engine().set(h_pad=0.04, hspace=0.02)

    return fig, axes