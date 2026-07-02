# internal imports
from data_acquisition.data_loader import load_subject_data
from processor.interpolator import interpolate_signal
from data_acquisition.data_visualizer import plot_interpolated_signal , plot_sensors_over_time , plot_synchronized_hd_signals
from processor.transfer_function import apply_transfer_functions
from synchronization.sync import synchronize_hd_acquisitions


path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos"
subject = "S2"

# read the data
subject_data = load_subject_data(path, subject)

pot_position = subject_data["potentiometer_position"]

non_condition_keys = {"potentiometer_position", "HI_values"}


for condition, content in subject_data.items():

    if condition in non_condition_keys:
        continue

    df = content["data"]

    params = content["params"]

    df_transferred = apply_transfer_functions(df, x0=pot_position)

    df_interpolated, info, mask = interpolate_signal(df_transferred)

    print(f"\nCondition: {condition}")

    # update dictionary with processed data
    subject_data[condition]["data"] = df_interpolated

    # plot signals
    #plot_sensors_over_time(params, df_interpolated, f"Subject: {subject} - {condition}")

    """
    #subject_data[condition]["mask"] = mask
    # plot to validate interpolation
    for column in df_interpolated.columns:
        plot_interpolated_signal(df_interpolated, mask, column, title=f"{subject} - {condition}")
    """

subject_data["HD"]["data"], subject_data["HD_EMG"]["data"] = (
    synchronize_hd_acquisitions(
        subject_data["HD"]["data"],
        subject_data["HD_EMG"]["data"],
        column_hd="gACCy",
        column_hd_emg="y"
    )
)

plot_synchronized_hd_signals(
    subject_data["HD"]["params"],
    subject_data["HD"]["data"],
    subject_data["HD_EMG"]["params"],
    subject_data["HD_EMG"]["data"],
    hd_accelerometer="gACCy",
    hd_emg_accelerometer="y",
    title=f"{subject} - Synchronization"
)





