import pandas as pd
import pingouin as pg

DXA_VARS_IN_GRAMS = {
    "Magro_total",
    "Massa_magra_braco_direito",
    "Massa_magra_braco_esquerdo",
    "Massa_magra_perna_direita",
    "Massa_magra_perna_esquerda",
    "Magro_tronco",
    "BMC_total",
}


def compute_icc_pairs(df: pd.DataFrame, variable_mapping: dict, icc_type: str = "ICC2") -> pd.DataFrame:
    """
    Compute Intraclass Correlation Coefficient (ICC) for each pair of variables
    defined in variable_mapping, comparing measurements obtained from two equipments (DXA vs TANITA).

    :param df: DataFrame containing both DXA and TANITA columns, one row per subject
    :param variable_mapping: dict mapping DXA column name -> TANITA column name
    :param icc_type: which ICC form to report (ICC1, ICC2, ICC3, ICC1k, ICC2k, ICC3k)
    :param alpha: significance level used for confidence intervals
    :return: DataFrame summarizing ICC, CI95%, p-value and interpretation for each pair
    """
    df = _convert_dxa_grams_to_kg(df, variable_mapping)

    results = []

    for dxa_col, tanita_col in variable_mapping.items():
        results.append(_compute_single_icc(df, dxa_col, tanita_col, icc_type))

    return pd.DataFrame(results)


def _convert_dxa_grams_to_kg(df: pd.DataFrame, variable_mapping: dict) -> pd.DataFrame:
    """
    Convert DXA columns known to be reported in grams (lean/muscle mass, BMC) to kg,
    so that they are on the same scale as the corresponding TANITA columns (kg).
    Returns a copy of the DataFrame with the relevant columns converted;
    columns not in DXA_VARS_IN_GRAMS, or not present in df, are left untouched.

    :param df: source DataFrame containing DXA columns
    :param variable_mapping: dict mapping DXA column name -> TANITA column name
    :return: copy of df with grams->kg conversion applied where relevant
    """
    df = df.copy()
    for dxa_col in variable_mapping.keys():
        if dxa_col in DXA_VARS_IN_GRAMS and dxa_col in df.columns:
            df[dxa_col] = df[dxa_col] / 1000
    return df



_ICC_TYPE_MAP = {
    "ICC1": "ICC(1,1)",
    "ICC2": "ICC(A,1)",
    "ICC3": "ICC(C,1)",
    "ICC1k": "ICC(1,k)",
    "ICC2k": "ICC(A,k)",
    "ICC3k": "ICC(C,k)",
}


def _compute_single_icc(df: pd.DataFrame, dxa_col: str, tanita_col: str, icc_type: str) -> dict:
    """
    Reshape a single pair of columns into long format and compute the ICC using pingouin.

    :param df: source DataFrame
    :param dxa_col: DXA variable name
    :param tanita_col: TANITA variable name
    :param icc_type: which ICC form to extract (accepts legacy names like 'ICC2' or new pingouin
                      labels like 'ICC(A,1)')
    :return: dict with ICC value, CI95%, p-value and interpretation
    """
    pair_df = df[[dxa_col, tanita_col]].dropna()
    n = len(pair_df)

    long_df = pd.DataFrame({
        "subject": list(pair_df.index) * 2,
        "rater": ["DXA"] * n + ["TANITA"] * n,
        "score": pd.concat([pair_df[dxa_col], pair_df[tanita_col]]).values
    })

    icc_table = pg.intraclass_corr(
        data=long_df, targets="subject", raters="rater", ratings="score"
    )
    icc_table["Type"] = icc_table["Type"].str.strip()
    icc_table = icc_table.set_index("Type")

    resolved_type = _ICC_TYPE_MAP.get(icc_type, icc_type)

    if resolved_type not in icc_table.index:
        return {
            "DXA_variable": dxa_col,
            "TANITA_variable": tanita_col,
            "n": n,
            "ICC": None,
            "CI95%": None,
            "p-value": None,
            "interpretation": None,
            "note": f"icc_type '{icc_type}' not found. Types available: {list(icc_table.index)}"
        }

    row = icc_table.loc[resolved_type]

    ci_col = next((c for c in icc_table.columns if "CI" in c.upper()), None)
    ci_value = tuple(round(v, 3) for v in row[ci_col]) if ci_col else None

    return {
        "DXA_variable": dxa_col,
        "TANITA_variable": tanita_col,
        "n": n,
        "ICC": round(row["ICC"], 3),
        "CI95%": ci_value,
        "p-value": "<0.001" if row["pval"] < 0.001 else round(row["pval"], 4),
        "interpretation": _interpret_icc(row["ICC"]),
        "note": None
    }


def _interpret_icc(icc_value: float) -> str:
    """
    Classify an ICC value into a qualitative agreement category, following Koo & Li (2016).

    :param icc_value: computed ICC value
    :return: qualitative interpretation string
    """
    if icc_value < 0.5:
        return "poor"
    elif icc_value < 0.75:
        return "moderate"
    elif icc_value < 0.9:
        return "good"
    else:
        return "excellent"