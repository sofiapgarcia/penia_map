"""
Functions to interactively select and apply trimming (cut) regions on signal data.

Available Functions
--------------------

[Public]
select_trim_regions(...): Interactively displays each condition (MVC, HI, HD)
    and lets the user select regions to cut. Saves the selected regions to a single
    Excel sheet (with a 'condition' column) in the subject's folder.
apply_trim_indices(...): Loads previously saved trim regions from Excel and removes
    the corresponding sample ranges from the data.

[Private]
_select_regions_for_condition(...): Displays a single interactive figure for one condition.
_save_trim_indices(...): Saves trim regions (all conditions) to a single Excel sheet.

[Private classes]
_TrimSelector: Holds the state (marked regions, plotted patches) for a single
    condition's interactive selection figure, and exposes the callback methods used
    by matplotlib (span selection and key press), avoiding nested functions.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def select_trim_regions(subject_data, subject_folder, non_condition_keys=None):
    """
    Interactively display one figure per condition (MVC, HI, HD) with one subplot
    per signal, allowing the user to select regions to cut using the mouse.

    Controls (per figure):
        - Click and drag on any subplot to mark a region to cut.
        - Press 'z' to remove the last marked region.
        - Close the figure window to confirm and move on to the next condition.

    :param subject_data: Dictionary as returned by load_subject_data (ideally already
        synced, so HD and HD_EMG are merged into a single "HD" entry)
    :param subject_folder: Path to the subject's folder, where trim_indices.xlsx will be saved
    :param non_condition_keys: Set of keys in subject_data to ignore
                               (default: {"potentiometer_position", "HI_values"})
    :return: Dictionary {condition: DataFrame with columns ['start', 'end']}
    """

    if non_condition_keys is None:
        non_condition_keys = {"potentiometer_position", "HI_values"}

    all_regions = {}

    for condition, content in subject_data.items():

        if condition in non_condition_keys:
            continue

        df = content["data"]

        signals = [(column, df[column]) for column in df.columns]

        regions = _select_regions_for_condition(condition, signals)
        all_regions[condition] = regions

    _save_trim_indices(all_regions, subject_folder)

    return all_regions


def apply_trim_indices(subject_data, subject_folder, non_condition_keys=None, verify_plot=True):
    """
    Load previously saved trim regions from 'trim_indices.xlsx' in subject_folder
    and remove the corresponding sample ranges from each condition's DataFrame
    (MVC, HI, HD).

    :param subject_data: Dictionary as returned by load_subject_data
    :param subject_folder: Path to the subject's folder containing trim_indices.xlsx
    :param non_condition_keys: Set of keys in subject_data to ignore
    :param verify_plot: If True, shows a before/after verification figure per condition
    :return: subject_data with trimmed dataframes (rows removed, index reset)
    """

    if non_condition_keys is None:
        non_condition_keys = {"potentiometer_position", "HI_values"}

    file_path = os.path.join(subject_folder, "trim_indices.xlsx")

    if not os.path.exists(file_path):
        print(f"No trim_indices.xlsx found ({file_path}). Skipping trimming.")
        return subject_data

    trim_df = pd.read_excel(file_path)

    for condition, content in subject_data.items():

        if condition in non_condition_keys:
            continue

        regions_df = trim_df[trim_df["condition"] == condition]

        if regions_df.empty:
            continue

        drop_ranges = list(zip(regions_df["start"], regions_df["end"]))

        df_before = content["data"]

        positions = np.arange(len(df_before))
        keep = np.ones(len(df_before), dtype=bool)

        for start, end in drop_ranges:
            keep &= ~((positions >= start) & (positions <= end))

        df_after = df_before.iloc[keep].reset_index(drop=True)

        if verify_plot:
            _plot_trim_verification(condition, df_before, df_after, drop_ranges)

        subject_data[condition]["data"] = df_after

    return subject_data

# ------------------------------------------------------------------------------------------------------------------- #
# private classes
# ------------------------------------------------------------------------------------------------------------------- #

class _TrimSelector:
    """
    Holds the interactive state for a single condition's trimming figure:
    the list of marked (start, end) regions and the corresponding axvspan
    patches drawn on each subplot. Provides callback methods for matplotlib
    events (span selection and key press) so no nested functions are needed.
    """

    def __init__(self, fig, axes, max_index):
        """
        :param fig: matplotlib Figure for this condition
        :param axes: array/list of matplotlib Axes (one per signal)
        :param max_index: Maximum valid sample index (i.e. len(signal) - 1);
            selected regions are clamped to [0, max_index]
        """
        self.fig = fig
        self.axes = axes
        self.max_index = max_index
        self.regions = []
        self.spans_patches = []

    def on_select(self, xmin, xmax):
        """
        Callback for SpanSelector: register a new region (clamped to the valid
        sample range) and draw it (as a red highlighted band) on every subplot.

        :param xmin: Left edge of the selected span (data coordinates)
        :param xmax: Right edge of the selected span (data coordinates)
        """
        start, end = int(round(xmin)), int(round(xmax))

        start = max(start, 0)
        end = min(end, self.max_index)

        if end <= start:
            return

        self.regions.append((start, end))

        patches = [ax.axvspan(start, end, color="red", alpha=0.3) for ax in self.axes]
        self.spans_patches.append(patches)

        self.fig.canvas.draw_idle()

    def on_key(self, event):
        """
        Callback for key press events: pressing 'z' removes the last
        marked region from every subplot.

        :param event: matplotlib KeyEvent
        """
        if event.key != "z" or not self.regions:
            return

        self.regions.pop()
        patches = self.spans_patches.pop()

        for patch in patches:
            patch.remove()

        self.fig.canvas.draw_idle()

    def get_regions_dataframe(self):
        """
        :return: DataFrame with columns ['start', 'end'], sorted by start
        """
        return (
            pd.DataFrame(self.regions, columns=["start", "end"])
            .sort_values("start")
            .reset_index(drop=True)
        )

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _plot_trim_verification(condition, df_before, df_after, drop_ranges):
    """
    Display a verification figure for one condition: one row per signal, with
    the "before" version (removed regions shaded in red) on the left column
    and the "after" (trimmed) version on the right column.

    :param condition: Name of the condition (e.g. "HD", "HI", "MVC")
    :param df_before: DataFrame with the signals before trimming
    :param df_after: DataFrame with the signals after trimming
    :param drop_ranges: List of (start, end) tuples that were removed
    """

    columns = df_before.columns
    n = len(columns)

    fig, axes = plt.subplots(n, 2, figsize=(12, 2.2 * n), squeeze=False)

    for row_idx, column in enumerate(columns):

        ax_before, ax_after = axes[row_idx, 0], axes[row_idx, 1]

        ax_before.plot(df_before[column].values, linewidth=0.8)
        for start, end in drop_ranges:
            ax_before.axvspan(start, end, color="red", alpha=0.3)
        ax_before.set_ylabel(column, fontsize=8)
        ax_before.tick_params(labelsize=7)

        ax_after.plot(df_after[column].values, linewidth=0.8, color="green")
        ax_after.tick_params(labelsize=7)

    axes[0, 0].set_title("Before", fontsize=9)
    axes[0, 1].set_title("After", fontsize=9)

    fig.suptitle(f"{condition} - trim verification")
    plt.tight_layout()
    plt.show()

def _select_regions_for_condition(condition, signals):
    """
    Display a single figure with one subplot per signal and let the user select
    regions to cut.

    :param condition: Name of the condition (e.g. "HD", "HI", "MVC")
    :param signals: List of (label, pandas Series) tuples to plot
    :return: DataFrame with columns ['start', 'end'], sorted by start
    """

    n = len(signals)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(14, 2.2 * n), squeeze=False)
    axes = axes[:, 0]

    max_index = max(len(series) for _, series in signals) - 1

    for ax, (label, series) in zip(axes, signals):
        ax.plot(series.values, linewidth=0.8)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Samples")
    fig.suptitle(
        f"{condition} - drag to mark regions to cut | 'z' = undo last | close window to confirm"
    )

    selector_state = _TrimSelector(fig, axes, max_index)

    # Keep references to the SpanSelectors, otherwise they get garbage collected
    span_selectors = [
        SpanSelector(
            ax, selector_state.on_select, "horizontal",
            useblit=True,
            props=dict(alpha=0.3, facecolor="red"),
            interactive=True,
        )
        for ax in axes
    ]

    plt.tight_layout()
    fig.canvas.mpl_connect("key_press_event", selector_state.on_key)
    plt.show(block=True)

    return selector_state.get_regions_dataframe()


def _save_trim_indices(all_regions, subject_folder):
    """
    Save trim regions for all conditions into a single Excel sheet, with a
    'condition' column identifying which condition each region belongs to.

    :param all_regions: Dictionary {condition: DataFrame with columns ['start', 'end']}
    :param subject_folder: Path to the subject's folder
    """

    file_path = os.path.join(subject_folder, "trim_indices.xlsx")

    combined_frames = []

    for condition, df in all_regions.items():

        df_with_condition = df.copy()
        df_with_condition.insert(0, "condition", condition)

        combined_frames.append(df_with_condition)

    if combined_frames:
        combined_df = pd.concat(combined_frames, ignore_index=True)
    else:
        combined_df = pd.DataFrame(columns=["condition", "start", "end"])

    combined_df.to_excel(file_path, index=False)

    print(f"Trim indices saved to: {file_path}")