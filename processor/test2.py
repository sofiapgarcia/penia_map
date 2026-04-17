import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\dataset.xlsx"

# read the data
target_column = "Estimated metabolic age"
df = pd.read_excel(filepath)



data = df[target_column].dropna()

bins = np.arange(data.min(), data.max() + 1, 1)

plt.hist(data, bins=bins, edgecolor="black")

plt.xlabel("Metabolic age")
plt.ylabel("Frequency")
plt.title("Histogram with bin size = 1")

plt.show()