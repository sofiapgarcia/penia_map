from data_preparation import analyze_excel_file , encode_categorical_columns
from ml_pipeline import feature_selection_filters

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\dataset.xlsx"

df = analyze_excel_file(filepath)
df_encoded = encode_categorical_columns(df)
df_features = df_encoded.drop(columns=["ID"], errors="ignore")
df = feature_selection_filters(df_features)

#print("oi")