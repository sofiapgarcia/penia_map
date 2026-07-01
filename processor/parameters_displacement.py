"""
Functions for Dynamic handgrip calibration analysis.

For each pos_XX.txt file (one grip position), this script:
  1. Reads the data with `extract_data_from_file`.
  2. Segments the signal into two moments: rest (not squeezing) and max squeeze.
  3. Computes the mean bits value for each moment.
  4. Shows an interactive plot with the segmentation marked, where you can
     drag over the max-squeeze region to mark extra intervals that should be
     excluded from the mean (not saved to disk).
  5. Aggregates everything into a summary table (one row per position).


Available Functions
-------------------
[Public]
function1(...):

[Private]
_function2(...):
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

# internal imports
from data_acquisition.data_loader import extract_data_from_file


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_PATH = r"C:\Users\Sofia\Desktop\Tese\teste Nmax e N0 HD"   # folder containing the pos_XX.txt files
SIGNAL_COL = "MICRO"                      # column to analyze (in bits)
SMOOTH_WINDOW = 51                        # rolling median window (samples)
TRANSITION_MARGIN_SAMPLES = 300           # number of samples excluded on each side of the
                                          # detected transition

SUMMARY_CSV = r"C:\Users\Sofia\Desktop\Tese\teste Nmax e N0 HD\handgrip_calibration_summary.csv"


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def analyze_handgrip_calibration(
    base_path,
    signal_col=SIGNAL_COL,
    smooth_window=SMOOTH_WINDOW,
    margin_samples=None,
):
    """
    Run the full calibration analysis for all position files in a folder.

    :param base_path: Path to the folder containing the pos_XX.txt files
    :param signal_col: Name of the column to analyze
    :param smooth_window: Rolling median window size used for segmentation
    :param margin_samples: Number of samples excluded on each side of the
        detected transition; defaults to `TRANSITION_MARGIN_SAMPLES` if not
        provided. Increase it for slower transitions, decrease it for
        faster ones.
    :return: DataFrame with one row per position (mean_rest_bits,
        mean_squeeze_bits, threshold_bits, n_manual_exclusions), sorted by
        position
    """
    files = sorted(glob.glob(os.path.join(base_path, "pos_*.txt")))

    if not files:
        raise FileNotFoundError(
            f"No pos_XX.txt files found in '{base_path}'."
        )

    results = []

    for file_path in files:
        _, params, df = extract_data_from_file(file_path)
        position = _extract_position(os.path.basename(file_path))
        sampling_rate = params["sampling rate"]

        res = _compute_moment_means(df, signal_col=signal_col, smooth_window=smooth_window,
                                    margin_samples=margin_samples)

        final = _plot_and_select_squeeze_exclusions(res, position, sampling_rate, signal_col=signal_col)

        results.append({
            "position": position,
            "mean_rest_bits": final["mean_rest"],
            "mean_squeeze_bits": final["mean_squeeze"],
            "threshold_bits": final["threshold"],
            "n_manual_exclusions": len(final["excluded_intervals"]),
        })

    summary_df = pd.DataFrame(results).sort_values("position").reset_index(drop=True)
    return summary_df

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _extract_position(filename):
    """
    Extract the position number from a file name (pos_XX).

    :param filename: File name to parse
    :return: Position identifier as a string
    """
    match = re.search(r"pos_(\d+)", filename)
    return match.group(1)


def _segment_two_states(signal, smooth_window=SMOOTH_WINDOW, margin_samples=None):
    """
    Segment a two-plateau signal into a first ("rest") and a second
    ("squeeze") state, based on chronological order. A margin
    of samples around the detected transition is marked as invalid, to
    avoid biasing the mean with transition values.

    :param signal: 1D array-like signal to segment
    :param smooth_window: Rolling median window size used for smoothing
    :param margin_samples: Number of samples excluded on each side of the
        detected transition.
    :return: Dict with is_squeeze (bool array, True = squeeze state),
        valid (bool array, False for samples in the transition margin),
        threshold and transition_idx
    """
    if margin_samples is None:
        margin_samples = TRANSITION_MARGIN_SAMPLES

    # Smooth signal to reduce noise
    smoothed = (
        pd.Series(signal)
        .rolling(window=smooth_window, center=True, min_periods=1)
        .median()
    )

    # Number of samples to consider in typical ranges (10% of the signal)
    n = len(smoothed)
    edge_n = max(1, int(n * 0.1))

    # Estimate typical rest and squeeze levels from signal edges
    start_level = smoothed.iloc[:edge_n].median()
    end_level = smoothed.iloc[-edge_n:].median()

    # Midpoint threshold between rest and squeeze
    threshold = (start_level + end_level) / 2

    # Squeeze mask
    is_squeeze_raw = smoothed.values < threshold

    # First point where squeeze starts
    transition_idx = int(np.argmax(is_squeeze_raw)) if is_squeeze_raw.any() else n // 2

    # Create final squeeze mask
    is_squeeze = np.zeros(n, dtype=bool)
    is_squeeze[transition_idx:] = True

    # Mask out transition region to avoid mixing states
    valid = np.ones(n, dtype=bool)
    lo = max(0, transition_idx - margin_samples)
    hi = min(n, transition_idx + margin_samples)
    valid[lo:hi] = False

    return {
        "is_squeeze": is_squeeze,
        "valid": valid,
        "threshold": threshold,
        "transition_idx": transition_idx,
    }


def _compute_moment_means(df, signal_col=SIGNAL_COL, smooth_window=SMOOTH_WINDOW,
                          margin_samples=None):
    """
    Compute the mean bits value for the two moments (rest and max squeeze).

    :param df: DataFrame containing the signal data
    :param signal_col: Name of the column to analyze
    :param smooth_window: Rolling median window size used for segmentation
    :param margin_samples: Number of samples excluded on each side of the
        detected transition; defaults to `TRANSITION_MARGIN_SAMPLES` if not
        provided
    :return: Dict with mean_rest, mean_squeeze, threshold, is_squeeze mask,
        valid mask and the raw signal array
    """
    signal = df[signal_col].values
    seg = _segment_two_states(signal, smooth_window, margin_samples)

    is_squeeze = seg["is_squeeze"]
    valid = seg["valid"]
    rest_mask = valid & ~is_squeeze
    squeeze_mask = valid & is_squeeze

    return {
        "mean_rest": signal[rest_mask].mean(),
        "mean_squeeze": signal[squeeze_mask].mean(),
        "threshold": seg["threshold"],
        "is_squeeze": is_squeeze,
        "valid": valid,
        "signal": signal,
        "squeeze_mask": squeeze_mask,
    }

def _plot_and_select_squeeze_exclusions(
    result,
    position,
    sampling_rate,
    signal_col=SIGNAL_COL
):
    """
    Show an interactive validation plot and allow manual exclusion of
    parts of the squeeze region from the mean calculation.

    Rest, squeeze, and transition regions are shaded.

    The user can:
      - Click and drag to exclude intervals inside the squeeze region
      - Press 'r' to reset all exclusions
      - Close the window to confirm selection

    :param result: Output dict from compute_moment_means
    :param position: Position identifier used in the plot title
    :param sampling_rate: Sampling rate in Hz (used for time axis)
    :param signal_col: Name of the analyzed signal column
    :return: Dict with updated mean_squeeze, mean_rest,
             threshold, and excluded_intervals
    """

    signal = result["signal"]
    is_squeeze = result["is_squeeze"]
    valid = result["valid"]
    squeeze_mask = result["squeeze_mask"]

    # Create time axis in seconds
    time = np.arange(len(signal)) / sampling_rate

    # Masks for visualization
    rest_shade = ~is_squeeze & valid
    squeeze_shade = is_squeeze & valid
    excluded_shade = ~valid

    # State container (replaces closures)
    state = {
        "exclusions": [],
        "patches": []
    }

    # ---------------------------------------------------------------------
    # Plot setup
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        time,
        signal,
        color="dimgray",
        linewidth=0.8,
        label=signal_col,
        zorder=3
    )

    ax.fill_between(
        time,
        signal.min(),
        signal.max(),
        where=rest_shade,
        color="steelblue",
        alpha=0.15,
        label="Rest"
    )

    ax.fill_between(
        time,
        signal.min(),
        signal.max(),
        where=squeeze_shade,
        color="tomato",
        alpha=0.15,
        label="Max squeeze"
    )

    ax.fill_between(
        time,
        signal.min(),
        signal.max(),
        where=excluded_shade,
        color="lightgray",
        alpha=0.4,
        label="Transition (excluded)"
    )

    # Mean reference lines
    rest_line = ax.axhline(
        result["mean_rest"],
        color="steelblue",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean rest = {result['mean_rest']:.1f} bits"
    )

    squeeze_line = ax.axhline(
        result["mean_squeeze"],
        color="tomato",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean squeeze = {result['mean_squeeze']:.1f} bits"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{signal_col} (bits)")
    ax.set_title(
        f"Position {position} — drag to exclude squeeze regions | "
        f"'r' to reset | close window to confirm"
    )

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    span = SpanSelector(
        ax,
        lambda tmin, tmax: _handle_selection(
            tmin, tmax, state, ax, signal, time, squeeze_mask, squeeze_line
        ),
        "horizontal",
        useblit=True,
        props=dict(alpha=0.25, facecolor="purple"),
        interactive=True,
        drag_from_anywhere=True,
    )

    fig.canvas.mpl_connect(
        "key_press_event",
        lambda event: _handle_key(event, state, ax, signal, time, squeeze_mask, squeeze_line)
    )

    plt.show()

    # Final recomputation after closing window
    final_mean = _recompute_squeeze_mean(signal, time, squeeze_mask, state["exclusions"])

    return {
        "mean_rest": result["mean_rest"],
        "mean_squeeze": final_mean,
        "threshold": result["threshold"],
        "excluded_intervals": state["exclusions"],
    }

def _recompute_squeeze_mean(signal, time, squeeze_mask, exclusions):
    """
    Recompute the squeeze mean after applying user-defined exclusions.

    :param signal: 1D signal array
    :param time: time axis (seconds)
    :param squeeze_mask: boolean mask for squeeze region
    :param exclusions: list of (t_start, t_end) intervals to exclude
    :return: updated squeeze mean (float or np.nan)
    """

    # Build mask of samples excluded by user
    user_excluded = np.zeros(len(signal), dtype=bool)

    for t0, t1 in exclusions:
        lo, hi = sorted((t0, t1))
        user_excluded |= (time >= lo) & (time <= hi)

    # Final valid squeeze samples
    final_mask = squeeze_mask & ~user_excluded

    if final_mask.any():
        return signal[final_mask].mean()

    return np.nan

def _handle_selection(tmin, tmax, state, ax, signal, time, squeeze_mask, squeeze_line):
    """
    Handle user selection of an exclusion interval.
    """

    if tmax - tmin <= 0:
        return

    # Store selected interval
    state["exclusions"].append((tmin, tmax))

    # Visual feedback (purple region)
    patch = ax.axvspan(tmin, tmax, color="purple", alpha=0.25, zorder=2)
    state["patches"].append(patch)

    # Recompute squeeze mean after exclusion
    new_mean = _recompute_squeeze_mean(signal, time, squeeze_mask, state["exclusions"])

    # Update plot
    squeeze_line.set_ydata([new_mean, new_mean])
    squeeze_line.set_label(f"Mean squeeze = {new_mean:.1f} bits (manual excl.)")

    ax.legend(loc="best", fontsize=8)
    ax.figure.canvas.draw_idle()

def _handle_key(event, state, ax, signal, time, squeeze_mask, squeeze_line):
    """
    Handle keyboard input ('r' resets exclusions).
    """

    if event.key != "r":
        return

    # Clear stored exclusions
    state["exclusions"].clear()

    # Remove all visual patches
    for patch in state["patches"]:
        patch.remove()

    state["patches"].clear()

    # Recompute mean with no exclusions
    new_mean = _recompute_squeeze_mean(signal, time, squeeze_mask, state["exclusions"])

    # Update plot
    squeeze_line.set_ydata([new_mean, new_mean])
    squeeze_line.set_label(f"Mean squeeze = {new_mean:.1f} bits")

    ax.legend(loc="best", fontsize=8)
    ax.figure.canvas.draw_idle()



if __name__ == "__main__":
    summary = analyze_handgrip_calibration(BASE_PATH, signal_col=SIGNAL_COL)
    print(summary)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSummary saved to: {SUMMARY_CSV}")