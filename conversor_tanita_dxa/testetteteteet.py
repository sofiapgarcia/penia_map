import matplotlib.pyplot as plt
import numpy as np
path1 = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\S100\S100_HD_EMG.txt"
path2 = r"C:\Users\Sofia\Desktop\TESTE\S444\S444_HD_EMG.txt"
path3 = r"C:\Users\Sofia\Desktop\opensignals_000780D8A877_2026-07-16_17-10-01.txt"
path4 = r"C:\Users\Sofia\Desktop\opensignals_0007804D3158_2026-07-16_17-23-07.txt"

data = np.loadtxt(path4)
plt.plot(data[:,2])
#plt.plot(data[:,3])
plt.show()
value1 = np.mean(data[2500:5000,2])
print(value1)
value2 = np.mean(data[10000:12000,2])
print(value2)