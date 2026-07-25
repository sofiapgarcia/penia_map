import os
import pandas as pd

from data_acquisition.data_loader import load_subject_data
from data_acquisition.data_visualizer import plot_sensors_over_time , plot_subject_data , plot_column_over_time
from processor.interpolator import interpolate_signal
from processor.transfer_function import apply_transfer_functions
from synchronization.sync_attempt2 import apply_saved_sync_indices, synchronize_hd_acquisitions
from processor.trimming import apply_trim_indices , select_trim_regions
from processor.emg import process_emg , get_emg_envelope
from processor.load_cell import process_force
from metrics.powerpenia.peak_power_calculation import process_hd_power
from metrics.emg_normalization import get_mvc_values , normalize_emg
from metrics.concentric_phase_detection_attempt2 import detect_concentric_eccentric_phases
from metrics.onset_detection import detect_emg_onsets
from constants import DISPLACEMENT_COLUMN
from metrics.powerpenia.coordination import get_muscle_coordination_info, build_coordination_dataframe


path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos"


non_condition_keys = {"handgrip_position", "handgrip_position_source","calibration","HI_values"}

#subjects = sorted(folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder)) and folder.startswith("S"))
subjects = ["S14"]
subjects_with_no_calibration = ["S1","S2", "S4","S8","S11","S12","S13","S14","S15","S16","S17","S18","S22","S24","S25","S84"]


# collects one power_info row (with subject id) per subject, for the final Excel export
all_power_info = []

for subject in subjects:

    subject_number = int(subject.replace("S", ""))

    if subject in ["S107","S53","S39","129","S130"]:
        continue
    #if subject in subjects_with_no_calibration:
        #continue

    print(f"\n{'='*60}")
    print(f"Processing {subject}")
    print(f"{'='*60}")

    subject_folder = os.path.join(path, subject)

    # Read data
    subject_data = load_subject_data(path, subject)

    handgrip_position = subject_data["handgrip_position"]
    hp_source = subject_data["handgrip_position_source"]

    calibration = subject_data.get("calibration")

    if calibration is not None:
        low_level = calibration["low_level"]
        high_level = calibration["high_level"]
    else:
        low_level = None
        high_level = None

    for condition, content in subject_data.items():

        if condition in non_condition_keys:
            continue

        df = content["data"]
        params = content["params"]
        fs = content["params"]["sampling rate"]

        # Apply transfer functions
        df_transferred = apply_transfer_functions(df, x0=handgrip_position, hp_source = hp_source , low_level = low_level , high_level = high_level)

        # Interpolate
        df_interpolated, info, mask = interpolate_signal(df_transferred)

        # Process emg signal
        df_processed = process_emg(df_interpolated, fs)

        # Process force signal
        df_processed = process_force(df_processed, fs)

        # Update dictionary
        subject_data[condition]["data"] = df_processed

        # plot the interpolated signal
        #for column in df_interpolated.columns:
            #plot_interpolated_signal(df_interpolated, mask, column)

        #if condition=="HD":
            #plot_sensors_over_time(params, df_processed)
            #plot_column_over_time(params, df_processed, DISPLACEMENT_COLUMN)

    # Synchronize acquisitions
    #subject_data_sync = synchronize_hd_acquisitions(subject_data,column_hd="gACCy",column_hd_emg="y",subject_folder=os.path.join(path, subject))
    subject_data_sync = apply_saved_sync_indices(subject_data, subject_folder , show_plot=False)
    #plot_subject_data(subject_data_sync)

    # Trim data
    # Interactively select and save trim regions for this subject
    #select_trim_regions(subject_data_sync, subject_folder, non_condition_keys)
    subject_data_sync = apply_trim_indices(subject_data_sync, subject_folder, non_condition_keys, verify_plot=False)
    #plot_subject_data(subject_data_sync)

    # concentric phase detection
    subject_data_phases = detect_concentric_eccentric_phases(subject_data_sync,show_plot=True)

    # peak power calculation
    subject_data_power = process_hd_power(subject_data_sync, show_plot=False)

    # calculate envelope
    subject_data_envelope = get_emg_envelope(subject_data_power, show_plot=False)

    # Get MVC values
    subject_data_mvc = get_mvc_values(subject_data_envelope, show_plot=False)

    # Normalize EMG signals
    subject_data_normalized = normalize_emg(subject_data_mvc, show_plot=False)

    # Onset information
    onset_hd , onset_hi = detect_emg_onsets(subject_data_normalized, show_pipeline_plots=True)

    # collect this subject's power_info row, tagged with their numeric id
    subject_power_info = subject_data_power["power_info"].copy()

    info_hd , info_hi = get_muscle_coordination_info(subject_data_normalized, onset_hd, subject_power_info, subject_id = subject_number)
    print("oi")
    #subject_power_info.insert(0, "subject_id", int(subject.replace("S", "")))
    #all_power_info.append(subject_power_info)

"""
# build the final table (one row per subject) and export it to Excel
power_summary = pd.concat(all_power_info, ignore_index=True)
power_summary_path = os.path.join(path, "power_domain_results.xlsx")
power_summary.to_excel(power_summary_path, index=False)
print(f"\nSaved power summary for {len(power_summary)} subject(s) to {power_summary_path}")
"""









