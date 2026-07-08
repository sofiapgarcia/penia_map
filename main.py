import os

from data_acquisition.data_loader import load_subject_data
from processor.interpolator import interpolate_signal
from processor.transfer_function import apply_transfer_functions
from synchronization.sync import synchronize_hd_acquisitions, apply_saved_sync_indices
from data_acquisition.data_visualizer import plot_subject_data, plot_interpolated_signal, plot_sensors_over_time
from processor.trimming import apply_trim_indices , select_trim_regions
from processor.emg import process_emg
from power.peak_power_calculation import process_hd_power

path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos"

non_condition_keys = {"potentiometer_position", "HI_values"}

#subjects = sorted(folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder)) and folder.startswith("S"))
subjects = ["S45"]
for subject in subjects:

    print(f"\n{'='*60}")
    print(f"Processing {subject}")
    print(f"{'='*60}")

    subject_folder = os.path.join(path, subject)

    # Read data
    subject_data = load_subject_data(path, subject)

    pot_position = subject_data["potentiometer_position"]

    for condition, content in subject_data.items():

        if condition in non_condition_keys:
            continue

        df = content["data"]
        params = content["params"]
        fs = content["params"]["sampling rate"]

        # Apply transfer functions
        df_transferred = apply_transfer_functions(df, x0=pot_position)

        # Interpolate
        df_interpolated, info, mask = interpolate_signal(df_transferred)

        # Process emg signal
        df_processed = process_emg(df_interpolated, fs, show_plot=False)


        # Update dictionary
        subject_data[condition]["data"] = df_processed

        # plot the interpolated signal
        #for column in df_interpolated.columns:
            #plot_interpolated_signal(df_interpolated, mask, column)

        #plot_sensors_over_time(params, df_interpolated)


    # Synchronize acquisitions
    #subject_data_sync = synchronize_hd_acquisitions(subject_data,column_hd="gACCy",column_hd_emg="y",subject_folder=os.path.join(path, subject))
    subject_data_sync = apply_saved_sync_indices(subject_data, subject_folder)
    #plot_subject_data(subject_data_sync)

    # Trim data
    # Interactively select and save trim regions for this subject
    #select_trim_regions(subject_data_sync, subject_folder, non_condition_keys)
    subject_data_sync = apply_trim_indices(subject_data_sync, subject_folder, non_condition_keys, verify_plot=False)
    #plot_subject_data(subject_data_sync)

    # peak power calculation
    subject_data_power = process_hd_power(subject_data_sync, show_plot=False)








