from files_validation import extract_data_from_file , plot_sensors_over_time
import matplotlib.pyplot as plt


filepath = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\S5\S5_HD_EMG.txt"
params , df = extract_data_from_file(filepath)


fig = plot_sensors_over_time(params, df)
plt.show()

print("oi")