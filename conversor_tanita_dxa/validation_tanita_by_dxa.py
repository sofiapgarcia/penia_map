import pandas as pd
from conversor_tanita_dxa.conversor_td import VARIABLE_MAPPING
from conversor_tanita_dxa.ICC import compute_icc_pairs
from conversor_tanita_dxa.bland_altman import plot_bland_altman_pairs


filepath = r"C:\Users\Sofia\Desktop\Tese\dados\dxa_tanita.xlsx"

# read the data
df = pd.read_excel(filepath, index_col=0)

# Calculate the ICC values
results_icc = compute_icc_pairs(df, VARIABLE_MAPPING, icc_type="ICC2")
print(results_icc.to_string(index=False))

# Plot the Bland Altman graphs with mapped variables
plot_bland_altman_pairs(df, VARIABLE_MAPPING, show_plot=True)