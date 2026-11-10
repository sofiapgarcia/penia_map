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

# internal imports
from constants import FS
from metrics.muscle_coordination_metrics import (_get_emg_peak_in_phase,_get_max_repetition_ihd)

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


def plot_missing_samples(df_interpolated, mask, fs=None, columns=["DISPLACEMENT"],
                         title="Missing samples", show=True):
    """
    Plot the RAW signals (without interpolation) and highlight in red
    the regions where samples were missing.

    The raw signal is rebuilt by putting NaN back where the mask is True,
    so the line is broken inside each gap and the red band marks it.

    :param df_interpolated: Interpolated DataFrame (output of interpolate_signal)
    :param mask: Boolean DataFrame, True where the sample was missing (output of interpolate_signal)
    :param fs: Sampling rate in Hz. If given, the x axis is in seconds, otherwise in samples
    :param columns: Signal columns to plot (default: all except nSeq)
    :param title: Figure title
    :param show: Whether to call plt.show()
    :return: Matplotlib figure
    """

    if columns is None:
        columns = [c for c in df_interpolated.columns if c != "nSeq"]

    # Raw signal: NaN where samples were missing
    df_raw = df_interpolated.mask(mask)

    # A row is "missing" if any signal column is NaN
    missing = mask[columns].any(axis=1).to_numpy()

    n = len(df_raw)
    scale = 1 / fs if fs else 1
    x = np.arange(n) * scale
    x_label = "Time (s)" if fs else "Samples"

    # Contiguous runs of missing samples -> (start, end) indices
    padded = np.concatenate(([0], missing.astype(int), [0]))
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1

    fig, axes = plt.subplots(
        len(columns), 1,
        figsize=(14, 2.5 * len(columns)),
        sharex=True,
        squeeze=False
    )
    axes = axes.flatten()

    for ax, col in zip(axes, columns):
        # Raw signal only (line breaks where there are NaN)
        ax.plot(x, df_raw[col].to_numpy(), color="tab:blue", lw=0.8)

        # Red highlight over each missing region
        for s, e in zip(starts, ends):
            ax.axvspan(
                (s - 0.5) * scale,
                (e + 0.5) * scale,
                facecolor="red", edgecolor="red",
                alpha=0.35, lw=1
            )

        ax.set_ylabel(col)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel(x_label)
    fig.suptitle(f"{title} — {int(missing.sum())} missing samples in {len(starts)} gap(s)")
    fig.tight_layout()

    if show:
        plt.show()

    return fig

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

import numpy as np
import matplotlib.pyplot as plt

# internal imports
from constants import FS
from metrics.muscle_coordination_metrics import (
    _get_emg_peak_in_phase,
    _get_max_repetition_ihd,
)


def _draw_double_arrow(ax, x1, x2, y, text, color, text_offset=0.02):
    """
    Draw a horizontal double-headed arrow between x1 and x2 with a label above it.

    :param ax: Matplotlib axes
    :param x1: Start x position
    :param x2: End x position
    :param y: Vertical position of the arrow
    :param text: Label shown above the arrow
    :param color: Arrow and label color
    :param text_offset: Vertical gap between arrow and label, as a fraction of the y-axis range
    """
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.8))

    # Offset the label upwards in axes-fraction units so the gap is the same
    # regardless of the data scale
    ax.annotate(text, xy=((x1 + x2) / 2, y),
                xytext=(0, 0), textcoords="offset points",
                xycoords="data",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=color,
                transform=ax.transData) if False else None

    y_min, y_max = ax.get_ylim()
    ax.text((x1 + x2) / 2, y + text_offset * (y_max - y_min), text,
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=color)


def plot_temporal_emg_metrics(
    subject_data_normalized,
    onset_df_dhd,
    onset_df_ihd,
    power_info,
    onset_extensor_column="EMG_Extensor_filt",
    onset_flexor_column="EMG_Flexor_filt",
    emg_extensor_column="EMG_Extensor_norm",
    emg_flexor_column="EMG_Flexor_norm",
    offset_column="offset_idx",
    ihd_values_key="IHD_values",
    margin=10,
    title="Métricas temporais da atividade muscular",
    show=True,
):
    """
    Illustrate the temporal muscle coordination metrics in a figure with two panels:
      - DHD: time difference between the EMG envelope peaks (normalized envelope,
        concentric phase, repetition at peak power)
      - IHD: onset time difference and co-contraction period (filtered EMG, the
        same signal used for onset/offset detection, repetition with the maximum
        IHD value). Extensor and flexor are shown in two stacked axes.

    All text shown in the figure (titles, labels, legends, annotations) is in Portuguese.

    :param subject_data_normalized: Dict output of normalize_emg
    :param onset_df_dhd: DataFrame output for the DHD condition from detect_emg_onsets
    :param onset_df_ihd: DataFrame output for the IHD condition from detect_emg_onsets
    :param power_info: DataFrame containing a 'rep_at_peak_power' column
    :param onset_extensor_column: Filtered extensor EMG column (also the muscle name in the onset DataFrames)
    :param onset_flexor_column: Filtered flexor EMG column (also the muscle name in the onset DataFrames)
    :param emg_extensor_column: Column name of normalized extensor envelope EMG (used for DHD peaks)
    :param emg_flexor_column: Column name of normalized flexor envelope EMG (used for DHD peaks)
    :param offset_column: Column name of the muscle offset sample index (IHD onset_df)
    :param ihd_values_key: Key holding the H1/H2/H3 values in subject_data_normalized
    :param margin: Fraction of the window duration added on each side for context
    :param title: Figure title
    :param show: Whether to call plt.show()
    :return: Tuple (matplotlib figure, dict with the metrics in ms)
    """

    color_ext, color_flex = "tab:blue", "tab:orange"
    metrics = {}

    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(2, 2)
    ax_dhd = fig.add_subplot(gs[:, 0])
    ax_ext = fig.add_subplot(gs[0, 1])
    ax_flex = fig.add_subplot(gs[1, 1], sharex=ax_ext)

    # ------------------------------------------------------------------ #
    # DHD: envelope peak difference
    # ------------------------------------------------------------------ #
    df_dhd = subject_data_normalized["DHD"]["data"]
    rep_dhd = int(power_info["rep_at_peak_power"].iloc[0])

    rep_data = onset_df_dhd[onset_df_dhd["rep"] == rep_dhd]
    ext_row = rep_data[rep_data["muscle"] == onset_extensor_column]

    conc_start = int(ext_row["concentric_start_idx"].values[0])
    conc_end = int(ext_row["concentric_end_idx"].values[0])
    duration = conc_end - conc_start

    win_start = max(int(conc_start - margin * duration), 0)
    win_end = min(int(conc_end + margin * duration), len(df_dhd) - 1)

    idx = np.arange(win_start, win_end + 1)
    t = (idx - win_start) / FS
    env_ext = df_dhd[emg_extensor_column].iloc[win_start:win_end + 1].to_numpy()
    env_flex = df_dhd[emg_flexor_column].iloc[win_start:win_end + 1].to_numpy()

    peak_idx_ext, peak_val_ext = _get_emg_peak_in_phase(df_dhd, conc_start, conc_end, emg_extensor_column)
    peak_idx_flex, peak_val_flex = _get_emg_peak_in_phase(df_dhd, conc_start, conc_end, emg_flexor_column)
    t_peak_ext = (peak_idx_ext - win_start) / FS
    t_peak_flex = (peak_idx_flex - win_start) / FS
    peak_diff_ms = (peak_idx_ext - peak_idx_flex) / FS * 1000

    ymax = max(np.nanmax(env_ext), np.nanmax(env_flex))

    ax_dhd.plot(t, env_ext, color=color_ext, lw=1.5, label="Envelope do extensor")
    ax_dhd.plot(t, env_flex, color=color_flex, lw=1.5, label="Envelope do flexor")
    ax_dhd.axvspan((conc_start - win_start) / FS, (conc_end - win_start) / FS,
                   color="grey", alpha=0.15, label="Fase concêntrica")

    ax_dhd.plot(t_peak_ext, peak_val_ext, "o", color=color_ext, ms=8, label="Pico do extensor")
    ax_dhd.plot(t_peak_flex, peak_val_flex, "o", color=color_flex, ms=8, label="Pico do flexor")
    ax_dhd.axvline(t_peak_ext, color=color_ext, ls=":", lw=1.2)
    ax_dhd.axvline(t_peak_flex, color=color_flex, ls=":", lw=1.2)

    _draw_double_arrow(ax_dhd, t_peak_ext, t_peak_flex, ymax * 1.08,
                       f"Diferença entre picos: {abs(peak_diff_ms):.0f} ms", "purple")

    ax_dhd.set_ylim(0, ymax * 1.5)
    ax_dhd.set_xlabel("Tempo (s)")
    ax_dhd.set_ylabel("Envelope de EMG normalizado")
    ax_dhd.set_title(f"DHD – Diferença entre picos (repetição {rep_dhd})")
    ax_dhd.grid(alpha=0.3)
    ax_dhd.legend(loc="upper right", fontsize=9)

    metrics["DHD"] = {"peak_diff_ms": peak_diff_ms}

    # ------------------------------------------------------------------ #
    # IHD: onset difference and co-contraction period (filtered EMG)
    # ------------------------------------------------------------------ #
    df_ihd = subject_data_normalized["IHD"]["data"]
    rep_ihd = _get_max_repetition_ihd(subject_data_normalized, ihd_values_key=ihd_values_key)

    rep_data = onset_df_ihd[onset_df_ihd["rep"] == rep_ihd]
    ext_row = rep_data[rep_data["muscle"] == onset_extensor_column]
    flex_row = rep_data[rep_data["muscle"] == onset_flexor_column]

    onset_ext = int(ext_row["onset_idx"].values[0])
    onset_flex = int(flex_row["onset_idx"].values[0])
    offset_ext = int(ext_row[offset_column].values[0])
    offset_flex = int(flex_row[offset_column].values[0])

    span_start = min(onset_ext, onset_flex)
    span_end = max(offset_ext, offset_flex)
    span = span_end - span_start

    win_start = max(int(span_start - margin * span), 0)
    win_end = min(int(span_end + margin * span), len(df_ihd) - 1)

    idx = np.arange(win_start, win_end + 1)
    t = (idx - win_start) / FS
    filt_ext = df_ihd[onset_extensor_column].iloc[win_start:win_end + 1].to_numpy()
    filt_flex = df_ihd[onset_flexor_column].iloc[win_start:win_end + 1].to_numpy()

    # Same symmetric y limits on both axes (filtered EMG oscillates around zero)
    m = max(np.nanmax(np.abs(filt_ext)), np.nanmax(np.abs(filt_flex)))
    y_low, y_high = -1.1 * m, 1.6 * m

    def _x(sample_idx):
        return (sample_idx - win_start) / FS

    ax_ext.plot(t, filt_ext, color=color_ext, lw=0.8, label="EMG filtrado do extensor")
    ax_flex.plot(t, filt_flex, color=color_flex, lw=0.8, label="EMG filtrado do flexor")

    # Onsets (dashed) and offsets (dotted) of both muscles on both axes
    for ax in (ax_ext, ax_flex):
        for onset, offset, color in [(onset_ext, offset_ext, color_ext),
                                     (onset_flex, offset_flex, color_flex)]:
            ax.axvline(_x(onset), color=color, ls="--", lw=1.4)
            ax.axvline(_x(offset), color=color, ls=":", lw=1.4)

    # Onset difference (top axis)
    onset_diff_ms = (onset_ext - onset_flex) / FS * 1000
    _draw_double_arrow(ax_ext, _x(onset_ext), _x(onset_flex), m * 1.10,
                       f"Diferença de onset: {abs(onset_diff_ms):.0f} ms", "black")

    # Co-contraction: from the last onset to the first offset (both axes shaded, arrow on bottom axis)
    cc_start = max(onset_ext, onset_flex)
    cc_end = min(offset_ext, offset_flex)
    cc_ms = max(cc_end - cc_start, 0) / FS * 1000

    if cc_end > cc_start:
        for ax in (ax_ext, ax_flex):
            ax.axvspan(_x(cc_start), _x(cc_end), color="tab:green", alpha=0.25)
        ax_flex.axvspan(0, 0, color="tab:green", alpha=0.25,
                        label=f"Período de co-contração ({cc_ms:.0f} ms)")
        _draw_double_arrow(ax_flex, _x(cc_start), _x(cc_end), m * 1.15,
                           f"Co-contração: {cc_ms:.0f} ms", "tab:green")
    else:
        ax_flex.plot([], [], " ", label="Sem período de co-contração")

    # Legend entries for the line styles
    ax_ext.plot([], [], color="grey", ls="--", label="Início de atividade (onset)")
    ax_ext.plot([], [], color="grey", ls=":", label="Fim de atividade (offset)")

    for ax in (ax_ext, ax_flex):
        ax.set_ylim(y_low, y_high)
        ax.set_ylabel("EMG filtrado")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    plt.setp(ax_ext.get_xticklabels(), visible=False)
    ax_flex.set_xlabel("Tempo (s)")
    ax_ext.set_title(f"IHD – Onset e co-contração (repetição {rep_ihd})")

    metrics["IHD"] = {
        "onset_diff_ms": onset_diff_ms,
        "cocontraction_ms": cc_ms,
    }

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if show:
        plt.show()

    return fig, metrics