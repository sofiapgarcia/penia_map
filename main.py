import os

# internal imports
from data_acquisition.data_loader import load_subject_data
from processor.interpolator import interpolate_signal
from processor.transfer_function import apply_transfer_functions
from synchronization.sync import synchronize_hd_acquisitions , apply_saved_sync_indices
from data_acquisition.data_visualizer import plot_subject_data

path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos"

subject = "S69"


print(f"\n{'='*60}")
print(f"Processing {subject}")
print(f"{'='*60}")

# Read data
subject_data = load_subject_data(path, subject)

pot_position = subject_data["potentiometer_position"]
subject_folder = os.path.join(path, subject)

non_condition_keys = {"potentiometer_position", "HI_values"}

for condition, content in subject_data.items():

    if condition in non_condition_keys:
        continue

    df = content["data"]
    params = content["params"]

    # Apply transfer functions
    df_transferred = apply_transfer_functions(df, x0=pot_position)

    # Interpolate
    df_interpolated, info, mask = interpolate_signal(df_transferred)

    print(f"  Condition: {condition}")

    # Update dictionary
    subject_data[condition]["data"] = df_interpolated

#subject_data_sync = apply_saved_sync_indices(subject_data, subject_folder)
#plot_subject_data(subject_data_sync)


# Synchronize acquisitions
subject_data_sync = synchronize_hd_acquisitions(
    subject_data,
    column_hd="gACCy",
    column_hd_emg="y",
    subject_folder=os.path.join(path, subject))






