from data_preparation import analyze_excel_file , encode_categorical_columns
from data_visualization import plot_distribution
from feature_reduction import *
import pandas as pd

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\dataset.xlsx"

# read the data

df = pd.read_excel(filepath)
duplicates = df.columns[df.columns.duplicated()]
print("Duplicated columns:", duplicates)

# identify missing values, duplicates and outliers
#df = analyze_excel_file(filepath)

# encode categorical columns
df_encoded = encode_categorical_columns(df)

# remove participant column ID
df_features = df_encoded.drop(columns=["ID"], errors="ignore")
duplicates = df_features.columns[df_features.columns.duplicated()]
print("Duplicated columns:", duplicates)
print("Columns in the DataFrame:")
for col in df_features.columns:
    print(col)

# plot distribution of one variable
#plot_distribution(df_features, "Estimated metabolic age")

# correlation with "Estimated metabolic age" column
corr = correlation_with_target(df_features, verbose=False)

# test normality of metabolic age column
test_normality(df_features, column="Estimated metabolic age")

# correlation summary by groups of variables
group_summary = group_correlation_summary(corr)

df_normalized = normalize_dataset(df_features, scaler_type="standard")

# Principal Component Analysis
# decide the suitable number of principal components
pca_full, cumulative = inspect_pca_variance(df_normalized)
n_components = 7
# apply pca
pca, X_pca_df, loadings = apply_pca(df_normalized, n_components)


print("oi")