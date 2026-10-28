"""
NHANES 2001-2002 -- Bone Mineral Density (BMD) Prediction Equation
==================================================================

What this script does:
  1. Reads the NHANES 2001-2002 files:
       - DEMO_B.XPT  -> Demographics
       - BMX_B.XPT   -> Body measurements
       - DXX_B.XPT   -> Whole-body DXA

  2. Merges the three files on the unique identifier SEQN.

  3. Selects the relevant anthropometry and body composition variables.

  4. Removes highly correlated variables using the
     adaptive_correlation_matrix function.

  5. When two variables are highly correlated, keeps the
     variable with the highest absolute correlation with BMD.

  6. Runs a bidirectional stepwise regression based on p-values.

  7. Computes VIF to assess multicollinearity in the final model.

  8. Prints and saves the final equation.
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# internal imports
from ml_pipeline.preprocessing.linear_regression_feature_reduction import adaptive_correlation_matrix
from reliability_analysis_tanita.run_reliability_analysis import generate_sample_characterization_table

# ------------------------------------------------------------------
# 1. LOAD FILES
# ------------------------------------------------------------------

DATA_DIR = r"C:\Users\Sofia\Downloads"

demo = pd.read_sas(os.path.join(DATA_DIR, "DEMO_B.xpt"), format="xport")
bmx = pd.read_sas(os.path.join(DATA_DIR, "BMX_B.xpt"), format="xport")
dxx = pd.read_sas(os.path.join(DATA_DIR, "DXX_B.xpt"), format="xport")

print(f"\nDXX_B raw rows (all 5 imputations per participant): {dxx.shape[0]}")

# ------------------------------------------------------------------
# 1b. KEEP ONLY ONE IMPUTATION PER PARTICIPANT (_MULT_ == 1)
# ------------------------------------------------------------------
# NHANES documentation, DXX_B codebook:
# "the data file contains 5 records for each survey participant...
#  Only 1 record should be used in calculating sample sizes."

if "_MULT_" in dxx.columns:
    dxx = dxx[dxx["_MULT_"] == 1].reset_index(drop=True)
else:
    raise KeyError(
        "_MULT_ column not found in DXX_B -- cannot deduplicate imputations. "
        "Check the file/column name before proceeding."
    )

print(f"DXX_B rows after keeping _MULT_ == 1 (one row per participant): {dxx.shape[0]}")

# ------------------------------------------------------------------
# 2. MERGE ON SEQN
# ------------------------------------------------------------------

df = demo.merge(bmx, on="SEQN", how="inner").merge(dxx, on="SEQN", how="inner")

print(f"\nCombined dataset: {df.shape[0]} participants, {df.shape[1]} variables")


# ------------------------------------------------------------------
# 3. VARIABLE MAPPING
# ------------------------------------------------------------------

rename_map = {
    # Demographics / Anthropometry
    "RIAGENDR": "Gender",
    "RIDAGEYR": "Age",
    "BMXHT": "Height",
    "BMXWT": "BodyMass",
    "BMXBMI": "BMI",

    # DXA -- fat
    "DXDRLPF": "LegFatR_Pct",
    "DXDLLPF": "LegFatL_Pct",

    # DXA -- lean mass
    "DXDTOLE": "GlobalMuscle_g",
    "DXDRALE": "ArmMuscleR_g",
    "DXDLALE": "ArmMuscleL_g",
    "DXDRLLE": "LegMuscleR_g",
    "DXDLLLE": "LegMuscleL_g",

    # DXA -- bone
    "DXDTOBMC": "BoneMass_g",

    # TARGET
    "DXDTOBMD": "BoneDensity",
}

available = [c for c in rename_map if c in df.columns]

df = df[["SEQN"] + available].rename(columns=rename_map)


# ------------------------------------------------------------------
# 4. CONVERT g -> kg
# ------------------------------------------------------------------

gram_columns = [
    "GlobalMuscle_g", "ArmMuscleR_g", "ArmMuscleL_g",
    "LegMuscleR_g", "LegMuscleL_g", "TorsoMuscle_g", "BoneMass_g",
]

for col in gram_columns:
    if col in df.columns:
        new_col = col.replace("_g", "Kg")
        df[new_col] = df[col] / 1000
        df.drop(columns=[col], inplace=True)


# ------------------------------------------------------------------
# 5. DATA CLEANING
# ------------------------------------------------------------------

# Whole-body DXA is valid for ages 8-59.
# Only adults are considered here.
df = df[df["Age"] >= 18]

model_vars = [
    "BoneDensity", "Gender", "Height", "BodyMass", "BMI",
    "LegFatR_Pct", "LegFatL_Pct",
    "GlobalMuscleKg", "ArmMuscleRKg", "ArmMuscleLKg",
    "LegMuscleRKg", "LegMuscleLKg",
    "BoneMassKg",
]
model_vars = [
    "BoneDensity", "Gender", "Height", "BodyMass", "BMI",
    "GlobalMuscleKg","BoneMassKg",
]
model_vars = [
    "BoneDensity", "Height", "BodyMass", "BMI",
    "GlobalMuscleKg","BoneMassKg","Age","Gender"
]

model_vars = [v for v in model_vars if v in df.columns]

df_model = df[model_vars].dropna().reset_index(drop=True)

generate_sample_characterization_table(df_model, age_col="Age", gender_col="Gender", bmi_col="BMI")
# ------------------------------------------------------------------
# 6. RECODE GENDER
# ------------------------------------------------------------------

# NHANES: 1 = Male, 2 = Female
# Model:   1 = Male, 0 = Female
if "Gender" in df_model.columns:
    df_model["Gender"] = df_model["Gender"].map({1: 1, 2: 0})

print(f"\nFinal regression sample: {df_model.shape[0]} participants")
print("Variables used:")
print(list(df_model.columns))

# ------------------------------------------------------------------
# 7. CORRELATION MATRIX
# ------------------------------------------------------------------

print("\n" + "=" * 78)
print("CORRELATION MATRIX")
print("=" * 78)

# The function already returns the DataFrame with all correlations
corr_matrix , _ , _ = adaptive_correlation_matrix(df_model)

print("\nCorrelation matrix:")
print(corr_matrix)

# ------------------------------------------------------------------
# 8. MULTICOLLINEARITY REMOVAL
# ------------------------------------------------------------------

CORR_THRESHOLD = 0.80

print("\n" + "=" * 78)
print("MULTICOLLINEARITY CONTROL")
print("=" * 78)
print(f"\nThreshold used: |r| >= {CORR_THRESHOLD}")

y = df_model["BoneDensity"]
X = df_model.drop(columns=["BoneDensity"])

# Correlation of each variable with BMD (from the matrix already computed)
target_corr = corr_matrix["BoneDensity"].drop("BoneDensity").abs()

# Matrix restricted to the predictors
predictor_corr = corr_matrix.drop(index="BoneDensity").drop(columns="BoneDensity").abs()

# Upper triangular matrix
upper = predictor_corr.where(
    np.triu(np.ones(predictor_corr.shape), k=1).astype(bool)
)

# Identify variables to drop
to_drop = set()
removed_pairs = []

for col in upper.columns:
    for row in upper.index:
        corr_value = upper.loc[row, col]

        if pd.notna(corr_value) and corr_value >= CORR_THRESHOLD:
            # If one of the variables was already removed, skip the pair
            if row in to_drop or col in to_drop:
                continue

            # Keep the variable with the highest absolute correlation with BoneDensity
            if target_corr[row] >= target_corr[col]:
                variable_to_keep, variable_to_remove = row, col
            else:
                variable_to_keep, variable_to_remove = col, row

            to_drop.add(variable_to_remove)
            removed_pairs.append({
                "Variable_1": row,
                "Variable_2": col,
                "Correlation": corr_value,
                "Kept": variable_to_keep,
                "Removed": variable_to_remove,
                "Kept_target_corr": target_corr[variable_to_keep],
                "Removed_target_corr": target_corr[variable_to_remove],
            })


# ------------------------------------------------------------------
# 9. REMOVAL RESULTS
# ------------------------------------------------------------------

if removed_pairs:
    removed_pairs_df = pd.DataFrame(removed_pairs)
    print("\nHighly correlated pairs:")
    print(removed_pairs_df.to_string(index=False))
else:
    removed_pairs_df = pd.DataFrame()
    print(f"\nNo variable pair had |r| >= {CORR_THRESHOLD}.")


# ------------------------------------------------------------------
# 10. REDUCED DATASET
# ------------------------------------------------------------------

X_reduced = X.drop(columns=sorted(to_drop))

print("\nRemoved variables:")
if to_drop:
    for var in sorted(to_drop):
        print(f"  - {var}")
else:
    print("  None")

print("\nVariables available after correlation control:")
print(list(X_reduced.columns))


# ------------------------------------------------------------------
# 11. STEPWISE REGRESSION
# ------------------------------------------------------------------

def stepwise_selection(X, y, threshold_in=0.05, threshold_out=0.10, verbose=True):
    """
    Bidirectional stepwise selection based on p-values.

    threshold_in:  maximum p-value to add a variable.
    threshold_out: minimum p-value to remove a variable.
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
                    print(f"  + adds    {best_feature:<16} (p = {best_pval:.6f})")

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
                    print(f"  - removes {worst_feature:<16} (p = {worst_pval:.6f})")

        if not changed:
            break

    return included


print("\nRunning stepwise selection...\n")
selected_features = stepwise_selection(X_reduced, y)

print("\nSelected variables:")
print(selected_features)


# ------------------------------------------------------------------
# 12. FINAL MODEL
# ------------------------------------------------------------------

final_model = sm.OLS(y, sm.add_constant(X_reduced[selected_features])).fit()

print("\n" + "=" * 78)
print("OLS REGRESSION RESULTS")
print("=" * 78)
print(final_model.summary())
print("=" * 78)


# ------------------------------------------------------------------
# 13. VIF
# ------------------------------------------------------------------

print("\n" + "=" * 78)
print("VARIANCE INFLATION FACTOR (VIF)")
print("=" * 78)

X_vif = sm.add_constant(X_reduced[selected_features].copy())

vif_data = [
    {"Variable": column, "VIF": variance_inflation_factor(X_vif.values, i)}
    for i, column in enumerate(X_vif.columns)
    if column != "const"
]

vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)

print(vif_df.to_string(index=False))


# ------------------------------------------------------------------
# 14. FINAL EQUATION
# ------------------------------------------------------------------

coeffs = final_model.params
equation = f"BoneDensity (g/cm2) = {coeffs['const']:.5f}"

for feat in selected_features:
    sign = "+" if coeffs[feat] >= 0 else "-"
    equation += f" {sign} {abs(coeffs[feat]):.5f} * {feat}"

print("\nFinal prediction equation:\n")
print(equation)
print(f"\nR2: {final_model.rsquared:.4f}")
print(f"Adjusted R2: {final_model.rsquared_adj:.4f}")
print(f"F-statistic: {final_model.fvalue:.4f}")
print(f"F-statistic p-value: {final_model.f_pvalue:.6e}")

