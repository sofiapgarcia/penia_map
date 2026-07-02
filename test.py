# internal imports
from data_acquisition.data_loader import load_subject_data
from processor.interpolator import interpolate_signal
from data_acquisition.data_visualizer import plot_interpolated_signal
from processor.transfer_function import apply_transfer_functions

path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos"
subject = "S2"
condition_to_test = "HD"

import matplotlib.pyplot as plt
import numpy as np

from data_acquisition.data_loader import extract_data_from_file
from processor.transfer_function import convert_displacement

# Caminho para o ficheiro OpenSignals
FILE = r"C:\Users\Sofia\Desktop\Tese\teste Nmax e N0 HD\pos_36.txt"

# Carregar ficheiro
subject, params, df = extract_data_from_file(FILE)

# Valor do potenciómetro usado na calibração
x0 = 3      # <-- coloca aqui o valor correspondente

df = df.rename(columns={"MICRO": "DISPLACEMENT"})

# Converter displacement
df = convert_displacement(df, x0)

# Vetor temporal
fs = params["sampling rate"]
t = np.arange(len(df)) / fs

# Plot
plt.figure(figsize=(12,4))
plt.plot(t, df["DISPLACEMENT"])
plt.xlabel("Time (s)")
plt.ylabel("Displacement (cm)")
plt.title("Converted displacement")
plt.grid(True)
plt.tight_layout()
plt.show()