from data_preparation import analyze_excel_file , encode_categorical_columns
from data_visualization import plot_distribution
from feature_reduction import spearman_correlation_with_target
import pandas as pd

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\dataset.xlsx"

# read the data

df = pd.read_excel(filepath)


# identify missing values, duplicates and outliers
#df = analyze_excel_file(filepath)

# encode categorical columns
df_encoded = encode_categorical_columns(df)

# remove participant column ID
df_features = df_encoded.drop(columns=["ID"], errors="ignore")

# plot distribution of one variable
#plot_distribution(df_features, "Estimated metabolic age")

# correlation with "Estimated metabolic age" column
corr = spearman_correlation_with_target(df_features)


print("oi")