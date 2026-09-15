import heapq
from itertools import combinations
from math import comb
from tqdm import tqdm
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ml_pipeline.preprocessing.feature_reduction import adaptive_correlation_matrix


# ---------------------------------------------------------------------------
# Correlation-based pruning (uses adaptive_correlation_matrix instead of Pearson)
# ---------------------------------------------------------------------------
EXCLUSIVE_GROUPS = [
    # Peak power
    [
        "peak_power",
        "peak_power_mass",
        "peak_power_height_squared",
        "force_at_peak_power",
        "velocity_at_peak_power",
        "max_force",
        "max_velocity"
    ],

    # Cocontraction HD
    [
        "rep_pp_hd_cocontraction_ms",
        "rep_pp_hd_cocontraction_normalized",
    ],
    [
        "mean_hd_cocontraction_ms",
        "mean_hd_cocontraction_normalized",
    ],

    # Cocontraction HI
    [
        "rep_max_hi_cocontraction_ms",
        "rep_max_hi_cocontraction_normalized",
    ],
    [
        "mean_hi_cocontraction_ms",
        "mean_hi_cocontraction_normalized",
    ],

    # Overlap HD
    [
        "rep_pp_hd_emg_overlap_area",
        "rep_pp_hd_emg_overlap_area_normalized",
    ],
    [
        "mean_hd_emg_overlap_area",
        "mean_hd_emg_overlap_area_normalized",
    ],

    # Overlap HI
    [
        "rep_max_hi_emg_overlap_area",
        "rep_max_hi_emg_overlap_area_normalized",
    ],
    [
        "mean_hi_emg_overlap_area",
        "mean_hi_emg_overlap_area_normalized",
    ],

    # Onset difference: signed vs absolute
    [
        "rep_pp_hd_onset_diff_ms",
        "rep_pp_hd_onset_diff_ms_abs",
    ],
    [
        "rep_max_hi_onset_diff_ms",
        "rep_max_hi_onset_diff_ms_abs",
    ],
    [
        "mean_hd_onset_diff_ms",
        "mean_hd_onset_diff_ms_abs",
    ],
    [
        "mean_hi_onset_diff_ms",
        "mean_hi_onset_diff_ms_abs",
    ],

    # Envelope delay
    [
        "rep_pp_hd_envelope_peak_delay_ms",
        "rep_pp_hd_envelope_peak_delay_ms_abs",
        "rep_pp_hd_normalized_envelope_peak_delay",
        "rep_pp_hd_normalized_envelope_peak_delay_abs",
    ],
    [
        "mean_hd_envelope_peak_delay_ms",
        "mean_hd_envelope_peak_delay_ms_abs",
        "mean_hd_normalized_envelope_peak_delay",
        "mean_hd_normalized_envelope_peak_delay_abs",
    ],
]
# Cada regra é uma lista de "slots" (conjuntos de features). A combinação
# é inválida se contiver pelo menos uma feature de CADA slot da regra
# simultaneamente.
FORBIDDEN_COOCCURRENCES = [

    # EMG peak HD — rep
    [
        {"rep_pp_hd_emg_peak_ratio"},
        {"rep_pp_hd_emg_peak_extensor", "rep_pp_hd_emg_peak_flexor"},
    ],
    # EMG peak HD — mean
    [
        {"mean_hd_emg_peak_ratio"},
        {"mean_hd_emg_peak_extensor", "mean_hd_emg_peak_flexor"},
    ],
    # EMG peak HI — rep
    [
        {"rep_max_hi_emg_peak_ratio"},
        {"rep_max_hi_emg_peak_extensor", "rep_max_hi_emg_peak_flexor"},
    ],
    # EMG peak HI — mean
    [
        {"mean_hi_emg_peak_ratio"},
        {"mean_hi_emg_peak_extensor", "mean_hi_emg_peak_flexor"},
    ],
    [
        {"rep_pp_hd_Median_Frequency_extensor","rep_pp_hd_Median_Frequency_flexor"},
        {"rep_pp_hd_Median_Frequency_ratio"}
    ],
    [
        {"rep_pp_hd_Peak_Frequency_extensor", "rep_pp_hd_Peak_Frequency_flexor"},
        {"rep_pp_hd_Peak_Frequency_ratio"}
    ],
    [
        {"rep_max_hi_Peak_Frequency_ratio"},
        {"rep_max_hi_Peak_Frequency_flexor","rep_max_hi_Peak_Frequency_extensor"},
    ],
    [
        {"rep_max_hi_Median_Frequency_ratio"},
        {"rep_max_hi_Median_Frequency_flexor","rep_max_hi_Median_Frequency_extensor"},
        ],
]


def _violates_forbidden_cooccurrences(combo_set: set, forbidden_cooccurrences: list) -> bool:
    """
    True if combo_set contains at least one feature from every slot of
    any rule in forbidden_cooccurrences (i.e. the forbidden trio/tuple
    is fully present at once).
    """
    return any(
        all(combo_set & slot for slot in rule)
        for rule in forbidden_cooccurrences
    )

def _build_group_lookup(pool: list, exclusive_groups: list) -> dict:
    """
    Map each feature in pool to the index of the exclusive group it
    belongs to. Features not part of any group are simply absent from
    the dict (treated as unrestricted).
    """
    lookup = {}
    for group_idx, group in enumerate(exclusive_groups):
        for feat in group:
            if feat in pool:
                lookup[feat] = group_idx
    return lookup


def _violates_exclusive_groups(combo, group_lookup: dict) -> bool:
    """
    True if combo contains more than one feature from the same
    exclusive group.
    """
    seen = set()
    for f in combo:
        g = group_lookup.get(f)
        if g is None:
            continue
        if g in seen:
            return True
        seen.add(g)
    return False

def _combo_is_valid(combo, group_lookup: dict, forbidden_cooccurrences: list) -> bool:
    if group_lookup and _violates_exclusive_groups(combo, group_lookup):
        return False
    if forbidden_cooccurrences:
        if _violates_forbidden_cooccurrences(set(combo), forbidden_cooccurrences):
            return False
    return True

def _prune_by_correlation(
    X: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    y: pd.Series,
    corr_threshold: float = 0.95,
    verbose: bool = False,
):
    """
    Fast pairwise-correlation pruning (no OLS fit). Removes near-duplicate
    features, keeping the feature that is more correlated with the target.
    Correlation values come from a precomputed adaptive (Pearson/Spearman)
    matrix rather than a plain Pearson correlation, so both the pair-wise
    relationships and feature-target correlations already account for each
    variable's distribution.

    :param X: DataFrame with the features
    :param corr_matrix: precomputed adaptive correlation matrix covering
        all columns of X and the target, as returned by
        adaptive_correlation_matrix
    :param y: Target Series. Used to decide which feature of a
        correlated pair to keep (the one more correlated with target)
    :param corr_threshold: absolute correlation above which a pair is
        considered redundant
    :param verbose: print each removal
    :return: (kept_features, dropped) where dropped is a list of
        (feature, correlation, reason)
    """
    remaining = list(X.columns)
    dropped = []

    while len(remaining) > 1:

        # Get the correlations between the remaining features
        corr_arr = corr_matrix.loc[remaining, remaining].abs().to_numpy(copy=True)
        np.fill_diagonal(corr_arr, 0.0)
        max_corr = corr_arr.max()
        if max_corr <= corr_threshold:
            break
        i, j = np.unravel_index(corr_arr.argmax(), corr_arr.shape)
        f1, f2 = remaining[i], remaining[j]

        # Get the correlation of each feature with the target
        corr_f1_y = abs(corr_matrix.loc[f1, y.name])
        corr_f2_y = abs(corr_matrix.loc[f2, y.name])

        # Drop the feature less correlated with the target
        drop = f1 if corr_f1_y < corr_f2_y else f2
        keep = f2 if drop == f1 else f1

        remaining.remove(drop)

        dropped.append(
            (drop, max_corr, f"corr. {max_corr:.3f} with {keep}")
        )

        if verbose:
            print(
                f"[CORR] Removing {drop:<20} "
                f"corr={max_corr:.3f} with {keep}"
            )

    return remaining, dropped


# ---------------------------------------------------------------------------
# Best-subset selection (exhaustive search up to k_max features)
# ---------------------------------------------------------------------------

def _fit_ols(X: pd.DataFrame, y: pd.Series, features: list):
    """
    Fit an OLS model on the given subset of features (with an intercept).

    :param X: DataFrame with all available features
    :param y: Target Series
    :param features: list of feature names to include in the model
    :return: fitted statsmodels OLS results object, or None if features is empty
    """
    return sm.OLS(y, sm.add_constant(X[features])).fit() if features else None


def _ols_fit_fast(X_arr: np.ndarray, y_arr: np.ndarray):
    """
    Fast OLS fit via numpy lstsq (no statsmodels overhead) — used inside
    the exhaustive search loop where speed matters more than diagnostics.

    :param X_arr: 2D array of features (no intercept column)
    :param y_arr: 1D array, target
    :return: (rss, n_params) where n_params includes the intercept
    """
    n = X_arr.shape[0]
    Xc = np.column_stack([np.ones(n), X_arr])
    coef, residuals, rank, _ = np.linalg.lstsq(Xc, y_arr, rcond=None)
    if residuals.size > 0:
        rss = residuals[0]
    else:
        rss = np.sum((y_arr - Xc @ coef) ** 2)
    return rss, Xc.shape[1]


def _score_from_rss(rss: float, n: int, n_params: int, criterion: str, y_arr: np.ndarray):
    """
    Compute aic/bic/r2_adj from RSS without refitting via statsmodels.
    Only used for relative comparison inside the search, so constant
    terms that don't affect ranking are dropped from aic/bic.

    :param rss: residual sum of squares
    :param n: number of samples
    :param n_params: number of parameters (incl. intercept)
    :param criterion: "aic", "bic", or "r2_adj"
    :param y_arr: 1D array, target (needed for r2_adj)
    :return: score value
    """
    rss = max(rss, 1e-12)
    if criterion == "r2_adj":
        tss = np.sum((y_arr - y_arr.mean()) ** 2)
        r2 = 1 - rss / tss
        return 1 - (1 - r2) * (n - 1) / (n - n_params)
    if criterion == "aic":
        return n * np.log(rss / n) + 2 * n_params
    return n * np.log(rss / n) + n_params * np.log(n)  # bic


def _model_diagnostics(
    X: pd.DataFrame,
    y: pd.Series,
    features: list,
    vif_threshold: float,
    alpha: float,
):
    """
    Refit a candidate subset with statsmodels and check whether every
    predictor clears both a significance and a multicollinearity bar.
    VIF is computed with statsmodels' variance_inflation_factor, which
    fits one auxiliary regression per feature against the others in the
    subset — cheap here because this is only called on a short list of
    top-scoring candidates, not on every combination in the search.

    :param X: DataFrame with all available features
    :param y: Target Series
    :param features: subset of features to fit and check
    :param vif_threshold: VIF at or above this value flags a feature as
        collinear with the rest of the subset
    :param alpha: significance level for the per-coefficient p-value check
    :return: (model, p_values, vif, passes) where p_values and vif are
        dicts keyed by feature name (intercept excluded), and passes is
        True only if every feature clears both checks
    """
    model = _fit_ols(X, y, features)
    if model is None:
        return None, {}, {}, False

    p_values = model.pvalues.drop("const", errors="ignore").to_dict()

    if len(features) > 1:
        Xc = sm.add_constant(X[features])
        # column 0 is the constant, so feature i sits at column i + 1
        vif = {
            f: variance_inflation_factor(Xc.values, i + 1)
            for i, f in enumerate(features)
        }
    else:
        vif = {features[0]: 1.0}

    passes = (
        all(p < alpha for p in p_values.values())
        and all(v < vif_threshold for v in vif.values())
    )
    return model, p_values, vif, passes


def best_subset_selection(
    X: pd.DataFrame,
    y: pd.Series,
    k_max: int = 10,
    criterion: str = "bic",
    corr_prescreen: float = 0.95,
    max_combinations: int = 20_000_000,
    vif_threshold: float = 5.0,
    alpha: float = 0.05,
    n_candidates_to_check: int = 25,
    verbose: bool = False,
):
    """
    Exhaustive best-subset selection: fits every combination of 1..k_max
    features (drawn from a correlation-pruned pool, using adaptive_correlation_matrix)
    and scores it with `criterion` using fast numpy lstsq. Instead of
    keeping only the single top-scoring combination, it keeps the
    `n_candidates_to_check` best and walks them from best to worst,
    refitting each with statsmodels and picking the first whose
    coefficients are all significant (p < alpha) and free of severe
    multicollinearity (VIF < vif_threshold). If none of the top
    candidates pass both checks, falls back to the single best-scoring
    subset and flags it as not passing.

    :param X: DataFrame with candidate features
    :param y: Target Series
    :param k_max: maximum subset size to try. Search size grows
        combinatorially with both the number of surviving features and
        k_max, so this is the main lever to keep the search tractable
    :param criterion: "aic", "bic", or "r2_adj"
    :param corr_prescreen: correlation threshold for the adaptive-correlation
        pruning phase run before the exhaustive search (essential with many
        features — this is what makes very wide feature sets tractable at all)
    :param max_combinations: safety limit; raises instead of hanging if the
        search space (sum of C(n_features, k) for k=1..k_max) exceeds this
    :param vif_threshold: VIF at or above this value flags a feature as
        collinear when screening candidates
    :param alpha: significance level for the per-coefficient p-value check
        when screening candidates
    :param n_candidates_to_check: how many top-scoring subsets (by
        `criterion`) to refit and check for significance/VIF before
        falling back to the single best-scoring one
    :param verbose: print pruning steps and search progress
    :return: dict with best_features, best_score, model (statsmodels OLS
        refit on best_features), p_values, vif, passes_diagnostics,
        n_combinations_tested, dropped_by_prescreen
    """
    corr_matrix, _, _ = adaptive_correlation_matrix(
        pd.concat([X, y.rename(y.name)], axis=1)
    )

    pool, dropped = _prune_by_correlation(
        X, corr_matrix, y=y, corr_threshold=corr_prescreen, verbose=verbose,
    )

    group_lookup = _build_group_lookup(pool, EXCLUSIVE_GROUPS) if EXCLUSIVE_GROUPS else {}

    n = len(y)
    n_features = len(pool)
    k_max_eff = min(k_max, n_features, n - 2)
    if k_max_eff < k_max and verbose:
        print(f"[BEST SUBSET] k_max reduced to {k_max_eff} (features={n_features}, n={n})")

    total_combinations = sum(comb(n_features, k) for k in range(1, k_max_eff + 1))
    if total_combinations > max_combinations:
        raise ValueError(
            f"{total_combinations:,} combinations ({n_features} features, "
            f"k_max={k_max_eff}) exceeds max_combinations={max_combinations:,}. "
            f"Tighten corr_prescreen, lower k_max, or set max_features_before_search."
        )
    if verbose:
        print(f"[BEST SUBSET] Searching {total_combinations:,} combinations "
              f"({n_features} features, k_max={k_max_eff})...")

    # "goodness" is always higher-is-better, so aic/bic (lower is better)
    # get sign-flipped and r2_adj (higher is better) stays as-is. This
    # lets the min-heap below hold the n_candidates_to_check BEST subsets
    # regardless of which criterion is used.
    goodness_sign = -1 if criterion in ("aic", "bic") else 1

    X_arr = X[pool].values
    y_arr = y.values
    col_idx = {f: i for i, f in enumerate(pool)}

    # Min-heap of (goodness, features) capped at n_candidates_to_check;
    # heap[0] is always the WORST of the candidates currently kept, so a
    # new entry only needs to beat it to get in.
    top_candidates = []
    combo_iter = (
        (k, combo)
        for k in range(1, k_max_eff + 1)
        for combo in combinations(pool, k)
    )
    for k, combo in tqdm(combo_iter, total=total_combinations, desc="Combos", unit="combo", leave=False):

        if not _combo_is_valid(combo, group_lookup, FORBIDDEN_COOCCURRENCES):
            continue
        idx = [col_idx[f] for f in combo]
        rss, n_params = _ols_fit_fast(X_arr[:, idx], y_arr)
        score = _score_from_rss(rss, n, n_params, criterion, y_arr)
        goodness = goodness_sign * score
        entry = (goodness, list(combo))
        if len(top_candidates) < n_candidates_to_check:
            heapq.heappush(top_candidates, entry)
        elif goodness > top_candidates[0][0]:
            heapq.heapreplace(top_candidates, entry)

    # Best-to-worst order for the walk below
    top_candidates.sort(key=lambda e: -e[0])

    chosen = None
    for goodness, feats in top_candidates:
        model, p_values, vif, passes = _model_diagnostics(
            X, y, feats, vif_threshold=vif_threshold, alpha=alpha,
        )
        if passes:
            chosen = (feats, goodness, model, p_values, vif, True)
            break

    if chosen is None:
        # Nothing among the top candidates is fully clean; fall back to
        # the single best-scoring subset and flag it as such rather than
        # silently returning a model with insignificant or collinear terms.
        goodness, feats = top_candidates[0]
        model, p_values, vif, _ = _model_diagnostics(
            X, y, feats, vif_threshold=vif_threshold, alpha=alpha,
        )
        chosen = (feats, goodness, model, p_values, vif, False)
        if verbose:
            print(
                f"[BEST SUBSET] WARNING: none of the top {len(top_candidates)} "
                f"candidates had p<{alpha} and VIF<{vif_threshold} for every "
                f"feature; returning the best-scoring subset anyway "
                f"(p_values={p_values}, vif={vif})"
            )

    best_features, goodness, model, p_values, vif, passes_diagnostics = chosen
    best_score = goodness_sign * goodness  # undo the sign flip for reporting

    if verbose:
        flag = "OK" if passes_diagnostics else "DOES NOT PASS p/VIF CHECK"
        print(f"[BEST SUBSET] Best: {best_features} ({criterion}={best_score:.4f}) [{flag}]")

    return {
        "best_features": best_features,
        "best_score": best_score,
        "model": model,
        "p_values": p_values,
        "vif": vif,
        "passes_diagnostics": passes_diagnostics,
        "n_combinations_tested": total_combinations,
        "dropped_by_prescreen": dropped,
    }


# ---------------------------------------------------------------------------
# Runner for multiple targets
# ---------------------------------------------------------------------------

def _resolve_allowed_features(
    target: str,
    feature_cols: list,
    target_cols: list,
    exclude_features: dict,
    exclude_features_all: list,
):
    """
    Build the list of features allowed as predictors for a given target.

    :param target: name of the current target column
    :param feature_cols: full candidate list of feature columns
    :param target_cols: list of all target columns being modeled
    :param exclude_features: dict {target: [features to exclude for
        that target only]}
    :param exclude_features_all: features to exclude from every model
    :return: (allowed_features, manually_excluded_features)
    """
    allowed = list(feature_cols)

    manual_excl = list(exclude_features_all) + list(exclude_features.get(target, []))
    allowed = [f for f in allowed if f not in manual_excl and f != target]

    return allowed, manual_excl


def run_for_multiple_targets(
    df: pd.DataFrame,
    feature_cols: list,
    target_cols: list,
    criterion: str = "r2_adj",
    corr_prescreen: float = 0.95,
    k_max: int = 5,
    max_combinations: int = 20_000_000,
    vif_threshold: float = 5.0,
    alpha: float = 0.05,
    n_candidates_to_check: int = 25,
    exclude_features: dict = None,
    exclude_features_all: list = None,
    verbose: bool = False,
):
    """
    Run best-subset feature selection for each target in target_cols.

    :param df: source DataFrame containing features and targets
    :param feature_cols: candidate feature columns
    :param target_cols: targets to model, one at a time
    :param criterion: "aic", "bic", or "r2_adj"
    :param corr_prescreen: adaptive-correlation threshold for pruning
        before the exhaustive search
    :param k_max: max subset size to try
    :param max_combinations: safety limit for the exhaustive search
    :param vif_threshold: VIF at or above this value flags a feature as
        collinear when screening candidate subsets
    :param alpha: significance level for the per-coefficient p-value check
        when screening candidate subsets
    :param n_candidates_to_check: how many top-scoring subsets per target
        to check for significance/VIF before falling back to the single
        best-scoring one
    :param exclude_features: {target: [features excluded for that target]}
    :param exclude_features_all: features excluded from every model
    :return: {target: {selected_features, model, r2, r2_adj, p_values,
        vif, passes_diagnostics, dropped_by_prescreen, excluded_manually}}
    """
    results = {}
    exclude_features = exclude_features or {}
    exclude_features_all = exclude_features_all or []

    for target in target_cols:
        allowed, manual_excl = _resolve_allowed_features(
            target, feature_cols, target_cols,
            exclude_features, exclude_features_all,
        )

        if verbose:
            print(f"\n=== Target: {target} (criterion={criterion}) ===")
            if manual_excl:
                print(f"Manually excluded: {manual_excl}")

        sub = df[allowed + [target]].dropna()
        print("n:", len(sub))

        search = best_subset_selection(
            sub[allowed], sub[target],
            k_max=k_max,
            criterion=criterion,
            corr_prescreen=corr_prescreen,
            max_combinations=max_combinations,
            vif_threshold=vif_threshold,
            alpha=alpha,
            n_candidates_to_check=n_candidates_to_check,
            verbose=verbose,
        )
        selected, model = search["best_features"], search["model"]

        results[target] = {
            "selected_features": selected,
            "model": model,
            "r2": model.rsquared if model is not None else None,
            "r2_adj": model.rsquared_adj if model is not None else None,
            "p_values": search["p_values"],
            "vif": search["vif"],
            "passes_diagnostics": search["passes_diagnostics"],
            "dropped_by_prescreen": search["dropped_by_prescreen"],
            "excluded_manually": manual_excl,
        }

    return results


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def results_to_dataframe(results: dict) -> pd.DataFrame:
    """One row per target: target, n_features, features, r2, r2_adj, max_vif, passes."""
    rows = [{
        "target": target,
        "n_features": len(info["selected_features"]),
        "features": ", ".join(info["selected_features"]),
        "r2": info["r2"],
        "r2_adj": info["r2_adj"],
        "max_vif": max(info["vif"].values()) if info["vif"] else None,
        "max_p_value": max(info["p_values"].values()) if info["p_values"] else None,
        "passes_diagnostics": info["passes_diagnostics"],
    } for target, info in results.items()]
    return pd.DataFrame(rows)


def export_results_to_excel(results: dict, path: str, decimals: int = 4):
    """
    Save a summary sheet (one row per target) plus one coefficients sheet
    per target into a single Excel file. Each coefficients sheet now also
    reports VIF alongside the p-value, so non-significant or collinear
    terms are visible directly next to the coefficient they belong to.

    :param results: output of run_for_multiple_targets
    :param path: output .xlsx path
    :param decimals: rounding applied to the coefficient tables
    """
    summary = results_to_dataframe(results).round(decimals)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)

        for target, info in results.items():
            model = info["model"]
            if model is None or not info["selected_features"]:
                continue
            coef_table = pd.DataFrame({
                "coef": model.params,
                "std_err": model.bse,
                "p_value": model.pvalues,
            })
            # VIF has no entry for the intercept
            coef_table["vif"] = pd.Series(info["vif"])
            coef_table = coef_table.round(decimals)
            # Excel sheet names are capped at 31 chars
            sheet_name = target[:31]
            coef_table.to_excel(writer, sheet_name=sheet_name)