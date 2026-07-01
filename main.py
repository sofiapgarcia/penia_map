# internal imports
from data_acquisition.data_loader import load_subject_data
from processor.interpolator import interpolate_signal
from data_acquisition.data_visualizer import plot_interpolated_signal

path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos"
subject = "S2"

# read the data
subject_data = load_subject_data(path, subject)

for condition, content in subject_data.items():

    df = content["data"]

    df_interpolated, info, mask = interpolate_signal(df)

    print(f"\nCondition: {condition}")
    print(f"Inserted samples: {info['inserted_samples']}")

    # update dictionary with processed data
    subject_data[condition]["data"] = df_interpolated
    subject_data[condition]["mask"] = mask

    # plot to validate interpolation
    for column in ["gACCz", "gACCy", "gACCx", "x", "y", "z", "EMG_Extensor", "EMG_Flexor"]:
        if column in df_interpolated.columns:
            plot_interpolated_signal(df_interpolated, mask, column, title=f"{subject} - {condition}")

