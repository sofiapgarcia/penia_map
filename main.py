import os

from data_acquisition.data_loader import load_subject_data
from processor.interpolator import interpolate_signal
from processor.transfer_function import apply_transfer_functions
from synchronization.sync import synchronize_hd_acquisitions, apply_saved_sync_indices
from data_acquisition.data_visualizer import plot_subject_data, plot_interpolated_signal, plot_sensors_over_time
from processor.trimming import apply_trim_indices , select_trim_regions

path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos"

non_condition_keys = {"potentiometer_position", "HI_values"}

#subjects = sorted(folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder)) and folder.startswith("S"))
subjects = ["S36"]
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

        # Apply transfer functions
        df_transferred = apply_transfer_functions(df, x0=pot_position)

        # Interpolate
        df_interpolated, info, mask = interpolate_signal(df_transferred)

        # Update dictionary
        subject_data[condition]["data"] = df_interpolated

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
    select_trim_regions(subject_data_sync, subject_folder, non_condition_keys)
    #subject_data_sync = apply_trim_indices(subject_data_sync, subject_folder, non_condition_keys)
    #plot_subject_data(subject_data_sync)







