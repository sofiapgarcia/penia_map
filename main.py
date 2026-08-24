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
from metrics.powerpenia.scalogram import get_muscle_scalogram_info
from constants import DISPLACEMENT_COLUMN
from metrics.powerpenia.coordination import get_muscle_coordination_metrics
from metrics.spectral_density import get_muscle_power_spectral_density_info
from utils import build_subject_feature_row
from metrics.validate_onsets import validate_onsets
from data_acquisition.data_visualizer import plot_interpolated_signal


path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos"


non_condition_keys = {"handgrip_position", "handgrip_position_source","calibration","HI_values"}

subjects = sorted(folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder)) and folder.startswith("S"))
#subjects = ["S21"]
subjects_with_missing_samples = ["S122","S133","S21","S36","S49"]
subjects_with_no_calibration = ["S4","S14","S17","S22","S24","S25","S84"]
subjects_with_signals_bad_quality = ["S21"]


features = []

for subject in subjects:

    subject_number = int(subject.replace("S", ""))

    if subject in ["S107","S53","S39","129","S130","S25","S84"]:
        continue
    if subject in subjects_with_no_calibration:
        continue

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

        # plot the interpolated signal
        #for column in df_interpolated.columns:
            #plot_interpolated_signal(df_interpolated, mask, column)

        # Process emg signal
        df_processed = process_emg(df_interpolated, fs)

        # Process force signal
        df_processed = process_force(df_processed, fs, show_plot=False)

        # Update dictionary
        subject_data[condition]["data"] = df_processed

        #if condition=="HD":
            #plot_sensors_over_time(params, df_processed)
            #plot_column_over_time(params, df_processed, DISPLACEMENT_COLUMN)


    # Synchronize acquisitions
    #subject_data_sync = synchronize_hd_acquisitions(subject_data,column_hd="gACCy",column_hd_emg="target",subject_folder=os.path.join(path, subject))
    subject_data_sync = apply_saved_sync_indices(subject_data, subject_folder , show_plot=False)
    #plot_subject_data(subject_data_sync)

    # Trim data
    # Interactively select and save trim regions for this subject
    #select_trim_regions(subject_data_sync, subject_folder, non_condition_keys)
    subject_data_sync = apply_trim_indices(subject_data_sync, subject_folder, non_condition_keys, verify_plot=False)
    #plot_subject_data(subject_data_sync)


    # concentric phase detection
    subject_data_phases = detect_concentric_eccentric_phases(subject_data_sync, show_plot=False)

    # peak power calculation
    power_info = process_hd_power(subject_data_sync, show_plot=False)

    # calculate envelope
    subject_data_envelope = get_emg_envelope(subject_data_phases, show_plot=False)

    # Get MVC values
    subject_data_mvc = get_mvc_values(subject_data_envelope, show_plot=False)

    # Normalize EMG signals
    subject_data_normalized = normalize_emg(subject_data_mvc, show_plot=False)

    # Onset information

    onset_hd , onset_hi = detect_emg_onsets(subject_data_normalized, show_plot=False, show_pipeline_plots=False, pipeline_max_reps=2, show_coarse_detection_plots=False)

    onset_hd , onset_hi = validate_onsets(subject_data_normalized, onset_hd, onset_hi, subject_number, output_dir=os.path.join(path, subject), show_validation_plot=False, force_review=False)

    coord_hd , coord_hi = get_muscle_coordination_metrics(subject_data_normalized, onset_hd, onset_hi, power_info)

    #scalogram_hd , scalogram_hi = get_muscle_scalogram_info(subject_data_normalized,onset_hd, onset_hi, power_info, show_scalogram_plot=False)

    band_power_hd , band_power_hi = get_muscle_power_spectral_density_info(subject_data_normalized, onset_hd, onset_hi, power_info)

    hi_values = subject_data_normalized.get("HI_values")

    subject_row = build_subject_feature_row(subject_number, power_info, coord_hd, coord_hi, band_power_hd, band_power_hi, hi_values)

    features.append(subject_row)

final_df = pd.concat(features, ignore_index=True)
final_df.to_excel(os.path.join(path, "features_power_dyna.xlsx"), index=False)
print(f"\nExportado dataframe final com {len(final_df)} sujeitos.")
















