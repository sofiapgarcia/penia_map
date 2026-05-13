"""
Functions to build, evaluate, and apply DXA prediction models
using TANITA body composition measurements with direct variable mapping.

The module implements:
- Direct variable mapping between TANITA and DXA
- Correlation-based feature validation (threshold: 0.6)
- Simple linear regression for mapped variables
- Best feature selection when multiple DXA variables map to same TANITA variable
- Model application and reporting

Available Functions
-------------------
[Public]
build_dxa_predictor(...): Build simple linear regression models based on mapped TANITA-DXA variable pairs.
predict_dxa(...): Predict DXA variables for new TANITA samples using previously trained models.
get_model_report(...): Generate a report of which variables were used and which were excluded.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# internal imports
from processor.feature_reduction import adaptive_correlation_matrix

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #

VARIABLE_MAPPING = {
    # DXA variable -> TANITA variable
    "Massa_total": "Body mass(kg)",
    "Massa Corporal": "Body mass(kg)",
    "BMI_DXA": "Body mass index (BMI)",
    "Perc_gordura_total": "Global fat %",
    "Perc_gordura_braco_direito": "Arm fat (right) %",
    "Perc_gordura_braco_esquerdo": "Arm fat (left) %",
    "Perc_gordura_perna_direita": "Leg fat (right) %",
    "Perc_gordura_perna_esquerda": "Leg fat (left) %",
    "Perc_gordura_tronco": "Torso fat %",
    "Magro_total": "Global muscle kg",
    "Massa_magra_braco_direito": "Arm muscle (right) kg",
    "Massa_magra_braco_esquerdo": "Arm muscle (left) kg",
    "Massa_magra_perna_direita": "Leg muscle (right) kg",
    "Massa_magra_perna_esquerda": "Leg muscle (left) kg",
    "Magro_tronco": "Torso muscle kg",
    "BMC_total": "Estimated bone mass kg",
    "VAT_area": "Visceral fat rating",
    "VAT_volume": "Visceral fat rating",
    "VAT_massa_gordura": "Visceral fat rating",
    "RMR": "Daily calorie intake (DCI)",
    "Altura": "Altura_testes"
}

CORRELATION_THRESHOLD = 0.6


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def build_dxa_predictor(
        df,
        variable_mapping=VARIABLE_MAPPING,
        corr_threshold=CORRELATION_THRESHOLD,
        alpha=0.05
):
    """
    Build simple linear regression models for DXA variables based on
    mapped TANITA variables with correlation validation.

    The function:
    - Uses direct DXA → TANITA mapping
    - Computes adaptive correlations (Pearson/Spearman) based on normality
    - Checks correlation between mapped pairs (must be >= corr_threshold)
    - Fits simple OLS regression models (1 feature + intercept)
    - Excludes DXA variables with low correlation

    :param df: Input DataFrame containing both TANITA and DXA variables
    :param variable_mapping: Dictionary mapping DXA variables to TANITA variables
    :param corr_threshold: Minimum correlation required for model inclusion (default: 0.6)
    :param alpha: Significance level for normality test in adaptive correlation
    :return: Dictionary containing:
        - "models": Dict with best model per DXA variable
        - "report": Dict with metadata on included/excluded variables
    """

    results = {
        "models": {},
        "report": {
            "included_tanita_vars": [],
            "excluded_dxa_vars": [],
            "models_created": [],
            "low_correlation_pairs": []
        }
    }

    # -----------------------------------
    # Compute adaptive correlation matrix
    # -----------------------------------
    corr_matrix, method_matrix = adaptive_correlation_matrix(df, alpha=alpha)

    # -----------------------------------
    # Process each DXA → TANITA mapping
    # -----------------------------------
    for dxa_var, tanita_var in variable_mapping.items():

        print(f"\n{'=' * 60}")
        print(f"Processing DXA variable: {dxa_var}")
        print(f"Mapped TANITA variable: {tanita_var}")
        print(f"{'=' * 60}")

        try:
            # Get correlation and method used
            corr = corr_matrix.loc[tanita_var, dxa_var]
            method = method_matrix.loc[tanita_var, dxa_var]

        except KeyError:
            # Variables not in correlation matrix
            print(f"EXCLUDED: Variables not available in data")
            results["report"]["excluded_dxa_vars"].append({
                "dxa_variable": dxa_var,
                "tanita_variable": tanita_var,
                "reason": "Variables not found or not numeric"
            })
            continue

        abs_corr = abs(corr)

        print(f"  Correlation: r = {corr:.4f} (|r| = {abs_corr:.4f})")
        print(f"  Method: {method}")

        # -----------------------------------
        # Check if correlation meets threshold
        # -----------------------------------
        if abs_corr < corr_threshold:
            print(f"EXCLUDED: Correlation {abs_corr:.4f} < threshold {corr_threshold}")
            results["report"]["excluded_dxa_vars"].append({
                "dxa_variable": dxa_var,
                "tanita_variable": tanita_var,
                "correlation": float(corr),
                "correlation_method": method,
                "reason": f"Correlation below threshold ({corr_threshold})"
            })
            results["report"]["low_correlation_pairs"].append({
                "tanita_var": tanita_var,
                "dxa_var": dxa_var,
                "correlation": float(corr),
                "method": method
            })
            continue

        # -----------------------------------
        # Prepare data and build regression
        # -----------------------------------
        valid_idx = df[[tanita_var, dxa_var]].notna().all(axis=1)
        X = df.loc[valid_idx, [tanita_var]].copy()
        y = df.loc[valid_idx, dxa_var].copy()

        # Add constant (intercept)
        X_const = sm.add_constant(X, has_constant='add')

        # Fit OLS regression
        model = sm.OLS(y, X_const).fit()

        # Store model
        results["models"][dxa_var] = {
            "model": model,
            "feature": tanita_var,
            "r2": float(model.rsquared),
            "adjusted_r2": float(model.rsquared_adj),
            "correlation": float(corr),
            "correlation_method": method,
            "n_samples": len(X)
        }

        # Track reporting
        results["report"]["included_tanita_vars"].append(tanita_var)
        results["report"]["models_created"].append({
            "dxa_variable": dxa_var,
            "tanita_variable": tanita_var,
            "correlation": float(corr),
            "correlation_method": method,
            "r2": float(model.rsquared),
            "n_samples": len(X)
        })

        print(f"INCLUDED: {tanita_var} → {dxa_var}")
        print(f"   Correlation: {abs_corr:.4f} ({method})")
        print(f"   R²: {model.rsquared:.4f}")
        print(f"   Samples: {len(X)}")

    # Remove duplicates from included_tanita_vars
    results["report"]["included_tanita_vars"] = list(
        set(results["report"]["included_tanita_vars"])
    )

    return results


def predict_dxa(results, tanita_df):
    """
    Predict DXA variables from TANITA samples.

    :param results: Output dictionary from build_dxa_predictor (contains "models" key)
    :param tanita_df: DataFrame containing TANITA variables
    :return: DataFrame with original TANITA columns and predicted DXA values
    """

    models = results["models"]
    predictions = {}

    for dxa_var, info in models.items():
        model = info["model"]
        feature = info["feature"]

        # Check if feature exists in input data
        if feature not in tanita_df.columns:
            print(f"Warning: Feature '{feature}' not found in input data. Skipping {dxa_var}")
            continue

        # Prepare data and make predictions
        X = tanita_df[[feature]].copy()
        X_const = sm.add_constant(X, has_constant='add')
        predictions[dxa_var] = model.predict(X_const)

    # Create DXA predictions DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Keep only selected TANITA variables
    tanita_keep = [
        "Daily calorie intake (DCI)",
        "Estimated metabolic age","ID"
    ]

    available_tanita = [col for col in tanita_keep if col in tanita_df.columns]
    tanita_subset = tanita_df[available_tanita].copy()

    # Combine DXA + selected TANITA variables
    result_df = pd.concat([predictions_df, tanita_subset], axis=1)

    return result_df


def get_model_report(results):
    """
    Generate a detailed report of model building results.

    :param results: Output dictionary from build_dxa_predictor
    :return: None (prints formatted report)
    """

    report = results["report"]
    models = results["models"]

    print("\n" + "=" * 70)
    print("DXA PREDICTOR MODEL REPORT")
    print("=" * 70)

    # Summary statistics
    print(f"\nSUMMARY")
    print(f"  Total DXA variables processed: {len(models) + len(report['excluded_dxa_vars'])}")
    print(f"  Models created: {len(models)}")
    print(f"  Excluded: {len(report['excluded_dxa_vars'])}")

    # Included TANITA variables
    print(f"\nINCLUDED TANITA VARIABLES ({len(report['included_tanita_vars'])})")
    for var in sorted(report["included_tanita_vars"]):
        print(f"  • {var}")

    # Models created
    print(f"\nMODELS CREATED ({len(report['models_created'])})")
    print(f"{'TANITA Variable':<35} {'DXA Variable':<25} {'r':<8} {'Method':<10} {'R²':<8} {'N':<6}")
    print("-" * 102)
    for model_info in sorted(report["models_created"], key=lambda x: x["tanita_variable"]):
        print(
            f"{model_info['tanita_variable']:<35} "
            f"{model_info['dxa_variable']:<25} "
            f"{model_info['correlation']:>7.4f} "
            f"{model_info['correlation_method']:<10} "
            f"{model_info['r2']:>7.4f} "
            f"{model_info['n_samples']:>5}"
        )

    # Excluded DXA variables
    if report["excluded_dxa_vars"]:
        print(f"\nEXCLUDED DXA VARIABLES ({len(report['excluded_dxa_vars'])})")
        for exc_info in report["excluded_dxa_vars"]:
            print(f"  • {exc_info['dxa_variable']}")
            if "correlation" in exc_info:
                method = exc_info.get("correlation_method", "N/A")
                print(f"    → Mapped TANITA: {exc_info['tanita_variable']} (r={exc_info['correlation']:.4f}, {method})")
            print(f"    → Reason: {exc_info['reason']}")

    # Low correlation pairs
    if report["low_correlation_pairs"]:
        print(f"\nLOW CORRELATION PAIRS (below threshold)")
        for pair in report["low_correlation_pairs"]:
            method = pair.get("method", "N/A")
            print(
                f"  • {pair['tanita_var']:<35} → {pair['dxa_var']:<25} "
                f"r = {pair['correlation']:>7.4f} ({method})"
            )

    print("\n" + "=" * 70 + "\n")

