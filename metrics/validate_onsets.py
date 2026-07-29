import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Disable the default navigation toolbar: we implement our own zoom/pan so
# that left-click (drag) and scroll are always free for navigation, and the
# right mouse button is always free to confirm a position.
matplotlib.rcParams['toolbar'] = 'None'

EMG_COLUMNS = ['EMG_Extensor_filt', 'EMG_Flexor_filt']
DEFAULT_VALIDATION_FILENAME = "onsets_validated.xlsx"


def _is_missing(value):
    """
    Check whether a value represents a missing index (None, or NaN as read
    back from Excel).

    :param value: Value to check.
    :return: True if the value should be treated as missing.
    """
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _window_bounds(signal_size, center_idx, margin_samples):
    """
    Compute the [start, end) sample bounds of an initial viewing window
    centered (as much as possible) around a reference index. This is only
    the STARTING view -- the user can freely zoom/pan afterward.

    :param signal_size: Length of the full signal.
    :param center_idx: Reference sample index to center the window on.
    :param margin_samples: Margin (samples) shown on each side of center_idx.
    :return: (start, end) tuple of sample indices, end exclusive.
    """
    start = max(0, center_idx - margin_samples)
    end = min(signal_size, center_idx + margin_samples)
    if end <= start:
        start, end = 0, signal_size
    return start, end


def _autoscale_y(ax, ydata, pad_frac=0.08):
    """
    Rescale the y-axis to fit the data currently visible in the x-range,
    with a small padding. Called after every zoom/pan so the signal always
    fills the plot nicely.

    :param ax: Matplotlib Axes to rescale.
    :param ydata: Full 1D array with the y data plotted on ax.
    :param pad_frac: Fraction of the visible data range added as padding.
    :return: None.
    """
    xlim = ax.get_xlim()
    lo = max(0, int(np.floor(xlim[0])))
    hi = min(ydata.size, int(np.ceil(xlim[1])) + 1)
    if hi <= lo:
        return
    visible = ydata[lo:hi]
    if visible.size == 0:
        return
    ymin, ymax = float(visible.min()), float(visible.max())
    pad = (ymax - ymin) * pad_frac if ymax > ymin else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)


def _review_index(raw, fs, ref_idx, label, title, fallback_center=None,
                   initial_margin_ms=1500):
    """
    Interactively review (and optionally correct) a single detected index
    (an onset or an offset), with free navigation:

      - Scroll the mouse wheel to zoom in/out (centered on the cursor).
      - Click and drag with the LEFT mouse button to pan left/right.
      - Click with the RIGHT mouse button at the desired position to confirm
        / set the index there -- this is the ONLY way to confirm an index.
      - Closing the window keeps the value as-is (the automatic detection if
        there was one, or leaves it missing).

    If ref_idx is missing (None/NaN, e.g. the automatic detection failed),
    the plot still opens so the position can be set manually: the initial
    view is centered on `fallback_center` (e.g. the paired onset/offset) if
    available, otherwise on the middle of the signal -- either way, you can
    freely zoom/pan to find the right spot.

    :param raw: 1D array with the raw (filtered) EMG signal for this muscle.
    :param fs: Sampling rate (Hz).
    :param ref_idx: Automatically detected index to review, or None/NaN if no
        detection was available.
    :param label: 'onset' or 'offset' (used in labels/messages only).
    :param title: Title shown above the plot.
    :param fallback_center: Index to center the initial view on when ref_idx
        is missing (e.g. the paired onset/offset for the same rep/muscle).
    :param initial_margin_ms: Half-width (ms) of the initial view window
        around the centering index. Purely a starting point -- zoom/pan
        freely afterward.
    :return: The validated index (int), which may be the same as ref_idx if
        kept, a newly (right-)clicked index, or None if it's still missing.
    """
    missing = _is_missing(ref_idx)

    if not missing:
        center = int(ref_idx)
    elif not _is_missing(fallback_center):
        center = int(fallback_center)
    else:
        center = raw.size // 2

    margin_samples = int(round((initial_margin_ms / 1000.0) * fs))
    start, end = _window_bounds(raw.size, center, margin_samples)

    t = np.arange(raw.size)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(t, raw, color='black', linewidth=0.8)

    if not missing:
        ax.axvline(ref_idx, color='tab:red', linestyle='-', alpha=0.85,
                   label=f'Detected {label}')
        ax.legend(loc='upper right')
        instruction = (
            "Scroll = zoom | Left-drag = pan | Right-click = confirm index | "
            "Close window = keep the red line"
        )
    else:
        instruction = (
            f"No automatic {label} detected -- Scroll = zoom | Left-drag = pan | "
            "Right-click = confirm index | Close window = leave it missing"
        )

    ax.set_xlim(start, end)
    _autoscale_y(ax, raw)
    ax.set_title(f"{title}\n{instruction}", fontsize=10)
    ax.set_xlabel('Samples')
    plt.tight_layout()

    result = {'idx': None if missing else int(ref_idx)}
    drag_state = {'press_x': None, 'press_xlim': None}

    def on_scroll(event):
        if event.inaxes != ax or event.xdata is None:
            return
        cur_xlim = ax.get_xlim()
        scale_factor = 0.8 if event.button == 'up' else 1.25
        width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        width = max(10, min(width, raw.size * 1.2))
        relx = (event.xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        new_xlim = (event.xdata - width * relx, event.xdata + width * (1 - relx))
        ax.set_xlim(new_xlim)
        _autoscale_y(ax, raw)
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            drag_state['press_x'] = event.xdata
            drag_state['press_xlim'] = ax.get_xlim()
        elif event.button == 3 and event.xdata is not None:
            new_idx = int(round(event.xdata))
            new_idx = max(0, min(raw.size - 1, new_idx))
            result['idx'] = new_idx
            print(f"  [{'set manually' if missing else 'corrected'}] {title} -- {label} -> {new_idx}")
            plt.close(fig)

    def on_motion(event):
        if drag_state['press_x'] is None or event.inaxes != ax or event.xdata is None:
            return
        dx = drag_state['press_x'] - event.xdata
        xlim = drag_state['press_xlim']
        ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
        _autoscale_y(ax, raw)
        fig.canvas.draw_idle()

    def on_release(event):
        if event.button == 1:
            drag_state['press_x'] = None
            drag_state['press_xlim'] = None

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    plt.show(block=True)

    if result['idx'] is None:
        print(f"  [{'still missing' if missing else 'kept'}] {title} -- {label}"
              + ('' if missing else f": {ref_idx}"))

    return result['idx']


def _sheet_exists(path, sheet_name):
    """
    Check whether a given sheet name exists in an Excel file.

    :param path: Path to the Excel file.
    :param sheet_name: Name of the sheet to look for.
    :return: True if the sheet exists, False otherwise (including if the
        file itself does not exist or cannot be read).
    """
    try:
        xls = pd.ExcelFile(path)
        return sheet_name in xls.sheet_names
    except Exception:
        return False


def _load_validated_onsets(output_file):
    """
    Load previously validated onsets/offsets from this subject's validation
    file.

    :param output_file: Path to the validation Excel file.
    :return: (hd_df, hi_df) tuple of DataFrames (empty if the file or the
        corresponding sheet doesn't exist yet).
    """
    if not os.path.exists(output_file):
        return pd.DataFrame(), pd.DataFrame()

    hd = pd.read_excel(output_file, sheet_name='HD') if _sheet_exists(output_file, 'HD') else pd.DataFrame()
    hi = pd.read_excel(output_file, sheet_name='HI') if _sheet_exists(output_file, 'HI') else pd.DataFrame()
    return hd, hi


def _save_validated_onsets(output_file, hd_df, hi_df):
    """
    Persist the validated onset/offset tables to this subject's validation
    Excel file, overwriting it.

    :param output_file: Path to the validation Excel file.
    :param hd_df: DataFrame with the subject's validated HD onsets.
    :param hi_df: DataFrame with the subject's validated HI onsets/offsets.
    :return: None.
    """
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        hd_df.to_excel(writer, sheet_name='HD', index=False)
        hi_df.to_excel(writer, sheet_name='HI', index=False)


def _shade_phases(ax, phase_values, fs):
    """
    Shade contiguous concentric ('Cx') and eccentric ('Ex') phase segments
    on an axis whose x-axis is in seconds, based on a per-sample 'Phase'
    column. Rest ('R') segments are left unshaded.

    :param ax: Matplotlib Axes to shade.
    :param phase_values: 1D array-like of per-sample phase labels ('R',
        'C<x>', 'E<x>'), same length as the plotted signal.
    :param fs: Sampling rate (Hz), used to convert sample indices to time.
    :return: None.
    """
    phases = np.asarray(phase_values)
    if phases.size == 0:
        return

    concentric_labeled = False
    eccentric_labeled = False

    start = 0
    for i in range(1, phases.size + 1):
        if i == phases.size or phases[i] != phases[start]:
            phase = phases[start]
            if isinstance(phase, str) and phase.startswith('C'):
                ax.axvspan(start / fs, i / fs, color='tab:blue', alpha=0.15,
                           label='Concentric' if not concentric_labeled else None)
                concentric_labeled = True
            elif isinstance(phase, str) and phase.startswith('E'):
                ax.axvspan(start / fs, i / fs, color='tab:orange', alpha=0.15,
                           label='Eccentric' if not eccentric_labeled else None)
                eccentric_labeled = True
            start = i


def _plot_condition_validation(df_signal, validated_df, fs, condition_label, columns,
                                phase_column=None):
    """
    Plot the whole raw signal for a condition (one subplot per muscle), with
    the validated onsets (and offsets, if present) marked as vertical lines.
    Free navigation is available, like in the individual review plots:

      - Scroll the mouse wheel to zoom in/out (centered on the cursor).
      - Click and drag with the LEFT mouse button to pan left/right.

    All subplots share the x-axis, so zooming/panning on one moves all of
    them together.

    :param df_signal: DataFrame with the full raw signal for this condition
        (one column per muscle).
    :param validated_df: DataFrame with the validated 'onset_idx' (and
        'offset_idx', if applicable) for this condition, one row per
        repetition/muscle.
    :param fs: Sampling rate (Hz), used to label the x-axis in seconds.
    :param condition_label: 'HD' or 'HI' (used in the title).
    :param columns: Names of the EMG columns to plot.
    :param phase_column: Name of a per-sample phase column in df_signal
        ('R'/'C<x>'/'E<x>') to shade as concentric/eccentric spans behind
        the signal. None (default) skips shading.
    :return: None.
    """
    present_columns = [col for col in columns if col in df_signal.columns]
    if not present_columns:
        return

    has_offset = 'offset_idx' in validated_df.columns
    phase_values = (
        df_signal[phase_column].to_numpy()
        if phase_column is not None and phase_column in df_signal.columns
        else None
    )

    # This plot is just for viewing, so use the normal matplotlib toolbar
    # (magnifying-glass zoom, pan, home) instead of the custom scroll/drag
    # navigation used in the individual review plots.
    previous_toolbar = matplotlib.rcParams['toolbar']
    matplotlib.rcParams['toolbar'] = 'toolbar2'
    try:
        fig, axes = plt.subplots(len(present_columns), 1,
                                  figsize=(14, 4 * len(present_columns)), sharex=True)
    finally:
        matplotlib.rcParams['toolbar'] = previous_toolbar
    if len(present_columns) == 1:
        axes = [axes]

    t = np.arange(len(df_signal)) / fs
    raws = [df_signal[col].to_numpy() for col in present_columns]

    for ax, col, raw in zip(axes, present_columns, raws):
        if phase_values is not None:
            _shade_phases(ax, phase_values, fs)

        ax.plot(t, raw, color='black', linewidth=0.6)

        col_rows = validated_df[validated_df['muscle'] == col] if 'muscle' in validated_df.columns else validated_df

        onset_labeled = False
        offset_labeled = False
        for _, row in col_rows.iterrows():
            onset_idx = row.get('onset_idx')
            if not _is_missing(onset_idx):
                ax.axvline(onset_idx / fs, color='tab:green', linestyle='-', alpha=0.8,
                           label='Onset' if not onset_labeled else None)
                onset_labeled = True
            if has_offset:
                offset_idx = row.get('offset_idx')
                if not _is_missing(offset_idx):
                    ax.axvline(offset_idx / fs, color='tab:red', linestyle='-', alpha=0.8,
                               label='Offset' if not offset_labeled else None)
                    offset_labeled = True

        ax.set_ylabel(col)
        if onset_labeled or offset_labeled or phase_values is not None:
            ax.legend(loc='upper right')

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f"{condition_label} -- validated " + ("onsets/offsets" if has_offset else "onsets"))
    plt.tight_layout()
    plt.show(block=True)


def validate_onsets(subject_data, onset_hd, onset_hi, subject, output_dir,
                     emg_columns=None, initial_margin_ms_hd=300,
                     initial_margin_ms_hi=4000, force_review=False,
                     show_validation_plot=False,
                     validation_filename=DEFAULT_VALIDATION_FILENAME):
    """
    Manually review (and correct) the automatically detected EMG onsets (HD)
    and onsets/offsets (HI) for a single subject, with free zoom/pan
    navigation, and persist the validated values to an Excel file kept in
    that subject's own output folder, so the manual review never has to be
    repeated once it's been done.

    Behaviour:
      - If a validation file already exists in output_dir with BOTH the 'HD'
        and 'HI' sheets populated (and force_review is False), the manual
        review is skipped entirely and the stored values are returned as-is.
      - Otherwise, one interactive plot per (repetition, muscle, onset/offset)
        is shown: scroll to zoom, left-drag to pan, right-click to confirm a
        new position (the only way to confirm an index), close the window to
        keep the value as-is. Once every row has been reviewed, the results
        are written to the validation file, overwriting any previous content.

    :param subject_data: Dict as produced/consumed by detect_concentric_eccentric_phases,
        used here to fetch the raw EMG signals and sampling rates for HD/HI.
    :param onset_hd: DataFrame with the automatically detected HD onsets, as
        returned by detect_emg_onsets (one row per repetition/muscle).
    :param onset_hi: DataFrame with the automatically detected HI onsets/offsets,
        as returned by detect_emg_onsets (one row per repetition/muscle).
    :param subject: Subject identifier (e.g. int or str), used only for
        log/print messages.
    :param output_dir: Directory where this subject's validation Excel file
        lives (typically the subject's own output folder).
    :param emg_columns: Names of the EMG columns to process. Defaults to EMG_COLUMNS.
    :param initial_margin_ms_hd: Half-width (ms) of the initial view window
        for HD onsets. Just a starting point -- zoom/pan freely afterward.
    :param initial_margin_ms_hi: Half-width (ms) of the initial view window
        for HI onsets/offsets. Just a starting point -- zoom/pan freely
        afterward.
    :param force_review: If True, re-do the manual review even if a
        validation file already exists (overwriting it).
    :param show_validation_plot: If True, after the validated onsets are
        obtained (whether freshly reviewed or loaded from file), show one
        static plot per condition (HD, HI) with the whole raw signal per
        muscle and the validated onsets (and offsets, for HI) marked as
        vertical lines.
    :param validation_filename: Name of the validation Excel file (kept
        inside output_dir).
    :return: (validated_hd, validated_hi) tuple of DataFrames, in the same
        format as onset_hd/onset_hi but with validated 'onset_idx' (and
        'offset_idx' for HI) values -- ready to be used by downstream
        metrics instead of the raw automatic detections.
    """
    columns = emg_columns if emg_columns is not None else EMG_COLUMNS
    output_file = os.path.join(output_dir, validation_filename)

    existing_hd, existing_hi = _load_validated_onsets(output_file)

    if not force_review and not existing_hd.empty and not existing_hi.empty:
        print(f"Subject {subject}: validated onsets already found in '{validation_filename}' -- skipping manual review.")
        existing_hd = existing_hd.reset_index(drop=True)
        existing_hi = existing_hi.reset_index(drop=True)
        if show_validation_plot:
            if not existing_hd.empty:
                _plot_condition_validation(subject_data['HD']['data'], existing_hd,
                                            subject_data['HD']['params']['sampling rate'], 'HD', columns,
                                            phase_column='Phase')
            if not existing_hi.empty:
                _plot_condition_validation(subject_data['HI']['data'], existing_hi,
                                            subject_data['HI']['params']['sampling rate'], 'HI', columns)
        return existing_hd, existing_hi

    print(f"\n{'-'*60}\nSubject {subject}: manual onset/offset validation\n{'-'*60}")

    # --- HD: only 'onset_idx' is a real detection; concentric start/end come
    #     from the Phase column and are not reviewed here. ---
    validated_hd = onset_hd.copy()
    if not validated_hd.empty:
        fs_hd = subject_data['HD']['params']['sampling rate']
        df_hd = subject_data['HD']['data']

        for i, row in validated_hd.iterrows():
            col = row['muscle']
            if col not in columns:
                continue
            raw = df_hd[col].to_numpy()
            title = f"HD -- {col} -- Rep {row['rep']}"
            # if onset detection failed, fall back to the concentric start
            # (always available for HD) to still center the initial view
            new_onset = _review_index(
                raw, fs_hd, row['onset_idx'], 'onset', title,
                fallback_center=row.get('concentric_start_idx'),
                initial_margin_ms=initial_margin_ms_hd,
            )
            validated_hd.at[i, 'onset_idx'] = new_onset

    # --- HI: both 'onset_idx' and 'offset_idx' are reviewed. ---
    validated_hi = onset_hi.copy()
    if not validated_hi.empty:
        fs_hi = subject_data['HI']['params']['sampling rate']
        df_hi = subject_data['HI']['data']

        for i, row in validated_hi.iterrows():
            col = row['muscle']
            if col not in columns:
                continue
            raw = df_hi[col].to_numpy()

            title_onset = f"HI -- {col} -- Rep {row['rep']}"
            # if onset detection failed, fall back to the (raw) offset to
            # still center the initial view
            new_onset = _review_index(
                raw, fs_hi, row['onset_idx'], 'onset', title_onset,
                fallback_center=row.get('offset_idx'),
                initial_margin_ms=initial_margin_ms_hi,
            )
            validated_hi.at[i, 'onset_idx'] = new_onset

            title_offset = f"HI -- {col} -- Rep {row['rep']}"
            # if offset detection failed, prefer the just-reviewed onset as
            # fallback (more accurate than the raw one), else the raw onset
            fallback_for_offset = new_onset if not _is_missing(new_onset) else row.get('onset_idx')
            new_offset = _review_index(
                raw, fs_hi, row['offset_idx'], 'offset', title_offset,
                fallback_center=fallback_for_offset,
                initial_margin_ms=initial_margin_ms_hi,
            )
            validated_hi.at[i, 'offset_idx'] = new_offset

    _save_validated_onsets(output_file, validated_hd, validated_hi)
    print(f"Subject {subject}: validated onsets saved to '{output_file}'.")

    if show_validation_plot:
        if not validated_hd.empty:
            _plot_condition_validation(subject_data['HD']['data'], validated_hd,
                                        subject_data['HD']['params']['sampling rate'], 'HD', columns,
                                        phase_column='Phase')
        if not validated_hi.empty:
            _plot_condition_validation(subject_data['HI']['data'], validated_hi,
                                        subject_data['HI']['params']['sampling rate'], 'HI', columns)

    return validated_hd, validated_hi