"""
Simplified Bone Mineral Density (BMD) Prediction Equation
===========================================================

Reads a single Excel file, extracts the BMD column and a set of
TANITA-style predictor variables, and runs a bidirectional stepwise
regression (p-value based) to obtain a prediction equation for BMD.
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ------------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------------

EXCEL_PATH = r"C:\Users\Sofia\Desktop\Tese\dados\dxa_tanita.xlsx"
SHEET_NAME = 0

TARGET_COL = "BMD"

# Excel column name -> internal variable name
rename_map = {
    "Body mass(kg)": "BodyMass",
    "Body mass index (BMI)": "BMI",
    "Global fat %": "GlobalFatPct",
    "Arm fat (right) %": "ArmFatR_Pct",
    "Arm fat (left) %": "ArmFatL_Pct",
    "Leg fat (right) %": "LegFatR_Pct",
    "Leg fat (left) %": "LegFatL_Pct",
    "Torso fat %": "TorsoFatPct",
    "Global muscle kg": "GlobalMuscleKg",
    "Arm muscle (right) kg": "ArmMuscleRKg",
    "Arm muscle (left) kg": "ArmMuscleLKg",
    "Leg muscle (right) kg": "LegMuscleRKg",
    "Leg muscle (left) kg": "LegMuscleLKg",
    "Torso muscle kg": "TorsoMuscleKg",
    "Estimated bone mass kg": "BoneMassKg",
    "Visceral fat rating": "VisceralFatRating",
    "Daily calorie intake (DCI)": "DCI",
    "Estimated metabolic age": "MetabolicAge",
    "Global body water %": "GlobalWaterPct",
    "Altura_testes": "Height",
}


# ------------------------------------------------------------------
# 2. LOAD FILE
# ------------------------------------------------------------------

df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

print(f"\nLoaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")


# ------------------------------------------------------------------
# 3. SELECT AND RENAME COLUMNS
# ------------------------------------------------------------------

available = [c for c in rename_map if c in df.columns]
missing = [c for c in rename_map if c not in df.columns]

if missing:
    print("\nColumns not found in the file (ignored):")
    for col in missing:
        print(f"  - {col}")

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in the Excel file.")

df_model = df[[TARGET_COL] + available].rename(columns=rename_map)


# ------------------------------------------------------------------
# 4. CLEANING
# ------------------------------------------------------------------

df_model = df_model.dropna().reset_index(drop=True)

print(f"\nFinal regression sample: {df_model.shape[0]} rows")
print("Variables used:")
print(list(df_model.columns))

y = df_model[TARGET_COL]
X = df_model.drop(columns=[TARGET_COL])


# ------------------------------------------------------------------
# 5. STEPWISE REGRESSION
# ------------------------------------------------------------------

def stepwise_selection(X, y, threshold_in=0.05, threshold_out=0.10, verbose=True):
    """
    Bidirectional stepwise selection based on p-values.

    :param X: predictor variables.
    :param y: target variable.
    :param threshold_in: maximum p-value to add a variable.
    :param threshold_out: minimum p-value to remove a variable.
    :param verbose: whether to print each add/remove step.
    :return: list of selected feature names.
    """

    included = []

    while True:
        changed = False

        # ---------------- FORWARD ----------------
        excluded = [c for c in X.columns if c not in included]
        new_pval = pd.Series(index=excluded, dtype=float)

        for col in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [col]])).fit()
            new_pval[col] = model.pvalues[col]

        if not new_pval.empty:
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print(f"  + adds    {best_feature:<18} (p = {best_pval:.6f})")

        # ---------------- BACKWARD ----------------
        if included:
            model = sm.OLS(y, sm.add_constant(X[included])).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()

            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                if verbose:
                    print(f"  - removes {worst_feature:<18} (p = {worst_pval:.6f})")

        if not changed:
            break

    return included


print("\nRunning stepwise selection...\n")
selected_features = stepwise_selection(X, y)

print("\nSelected variables:")
print(selected_features)


# ------------------------------------------------------------------
# 6. VIF-BASED MULTICOLLINEARITY CONTROL
# ------------------------------------------------------------------

VIF_THRESHOLD = 5.0


def compute_vif(X):
    """
    Compute the Variance Inflation Factor for each column of X.

    :param X: predictor variables (without constant).
    :return: DataFrame with one row per variable and its VIF, sorted
        from highest to lowest VIF.
    """

    X_vif = sm.add_constant(X)

    vif_data = [
        {"Variable": column, "VIF": variance_inflation_factor(X_vif.values, i)}
        for i, column in enumerate(X_vif.columns)
        if column != "const"
    ]

    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False).reset_index(drop=True)


def reduce_by_vif(X, features, threshold=VIF_THRESHOLD, verbose=True):
    """
    Iteratively remove the variable with the highest VIF until every
    remaining variable is below the threshold (or only one is left).

    :param X: full predictor DataFrame.
    :param features: list of candidate feature names (from stepwise selection).
    :param threshold: maximum acceptable VIF.
    :param verbose: whether to print each removal step.
    :return: list of feature names with VIF < threshold.
    """

    features = list(features)

    while len(features) > 1:
        vif_df = compute_vif(X[features])
        worst_vif = vif_df.loc[0, "VIF"]

        if worst_vif <= threshold:
            break

        worst_feature = vif_df.loc[0, "Variable"]
        features.remove(worst_feature)

        if verbose:
            print(f"  - removes {worst_feature:<18} (VIF = {worst_vif:.2f})")

    return features


print(f"\nReducing multicollinearity (VIF threshold = {VIF_THRESHOLD})...\n")
final_features = reduce_by_vif(X, selected_features)

print("\nFinal variables after VIF control:")
print(final_features)


# ------------------------------------------------------------------
# 7. FINAL MODEL
# ------------------------------------------------------------------

final_model = sm.OLS(y, sm.add_constant(X[final_features])).fit()

print("\n" + "=" * 78)
print("OLS REGRESSION RESULTS")
print("=" * 78)
print(final_model.summary())
print("=" * 78)


# ------------------------------------------------------------------
# 8. VIF (FINAL CHECK)
# ------------------------------------------------------------------

print("\n" + "=" * 78)
print("VARIANCE INFLATION FACTOR (VIF)")
print("=" * 78)

vif_df = compute_vif(X[final_features])
print(vif_df.to_string(index=False))


# ------------------------------------------------------------------
# 9. FINAL EQUATION
# ------------------------------------------------------------------

coeffs = final_model.params
equation = f"{TARGET_COL} = {coeffs['const']:.5f}"

for feat in final_features:
    sign = "+" if coeffs[feat] >= 0 else "-"
    equation += f" {sign} {abs(coeffs[feat]):.5f} * {feat}"

print("\nFinal prediction equation:\n")
print(equation)
print(f"\nR2: {final_model.rsquared:.4f}")
print(f"Adjusted R2: {final_model.rsquared_adj:.4f}")