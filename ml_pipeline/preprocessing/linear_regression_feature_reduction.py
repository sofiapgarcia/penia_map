import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------------------------------------------------------
# Collinearity (VIF)
# ---------------------------------------------------------------------------

def compute_vif(X: pd.DataFrame) -> pd.Series:
    """
    Compute the VIF (Variance Inflation Factor) for each column of X.

    :param X: DataFrame with the features to evaluate
    :return: Series of VIF values per feature, sorted descending
        (VIF<5 ok, 5-10 moderate collinearity, >10 problematic)
    """
    Xc = sm.add_constant(X)
    vifs = pd.Series(
        [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])],
        index=X.columns,
    )
    return vifs.sort_values(ascending=False)


def _prune_by_correlation(
    X: pd.DataFrame,
    y: pd.Series = None,
    corr_threshold: float = 0.95,
    max_features: int = None,
    verbose: bool = False,
):
    """
    Fast pairwise-correlation pruning (no OLS fit). Removes near-duplicate
    features and, if there are more features than samples, cuts down to
    max_features - needed since VIF via OLS is not well-defined in that
    regime.

    :param X: DataFrame with the features
    :param y: Target Series (optional). Used to decide which feature of a
        correlated pair to keep (the one more correlated with y)
    :param corr_threshold: absolute correlation above which a pair is
        considered redundant
    :param max_features: optional cap on the number of features to keep
    :param verbose: print each removal
    :return: (kept_features, dropped) where dropped is a list of
        (feature, corr_or_None, reason)
    """
    remaining = list(X.columns)
    dropped = []

    while len(remaining) > 1:
        if max_features is not None and len(remaining) <= max_features:
            break

        corr = X[remaining].corr().abs().values.copy()
        np.fill_diagonal(corr, 0.0)
        max_val = corr.max()

        if max_val <= corr_threshold:
            if max_features is None:
                break
            # no more strongly correlated pairs, but still above max_features
            if y is not None:
                corr_with_y = X[remaining].corrwith(y).abs()
                worst = corr_with_y.idxmin()
                reason = f"cut for max_features (corr. with y={corr_with_y[worst]:.3f})"
            else:
                worst = remaining[-1]
                reason = "cut for max_features (arbitrary, no y given)"
            remaining.remove(worst)
            dropped.append((worst, None, reason))
            if verbose:
                print(f"[CORR] Removing {worst:<20} {reason}")
            continue

        # drop one feature from the most correlated pair
        i, j = np.unravel_index(np.argmax(corr), corr.shape)
        f1, f2 = remaining[i], remaining[j]
        if y is not None:
            drop = f1 if abs(X[f1].corr(y)) < abs(X[f2].corr(y)) else f2
        else:
            drop = f2
        keep = f2 if drop == f1 else f1

        remaining.remove(drop)
        dropped.append((drop, max_val, f"corr. {max_val:.3f} with {keep}"))
        if verbose:
            print(f"[CORR] Removing {drop:<20} corr={max_val:.3f}")

    return remaining, dropped


def filter_by_vif(
    X: pd.DataFrame,
    y: pd.Series = None,
    threshold: float = 5.0,
    corr_prescreen: float = 0.95,
    max_features_before_vif: int = None,
    verbose: bool = False,
):
    """
    Iteratively remove collinear features until every VIF is below
    threshold. Runs a fast correlation-based pruning phase first
    (_prune_by_correlation), then the real VIF filter via OLS on the
    reduced set.

    :param X: DataFrame with the features
    :param y: Target Series (optional). If given, activates target-aware
        mode: at each step the feature with the lowest correlation with y
        is removed (recommended); otherwise the highest-VIF feature is
    :param threshold: maximum acceptable VIF
    :param corr_prescreen: correlation threshold for the fast pruning
        phase. Use None to disable it (not recommended with many features)
    :param max_features_before_vif: cap on features entering the VIF phase.
        Recommended for wide datasets (features >= samples), since VIF is
        not well-defined otherwise
    :param verbose: print each removal
    :return: (kept_features, dropped), dropped covering both phases
    """
    remaining = list(X.columns)
    dropped = []

    if corr_prescreen is not None or max_features_before_vif is not None:
        remaining, dropped = _prune_by_correlation(
            X[remaining], y=y,
            corr_threshold=corr_prescreen if corr_prescreen is not None else 0.999999,
            max_features=max_features_before_vif,
            verbose=verbose,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        while len(remaining) > 1:
            vifs = compute_vif(X[remaining])
            candidates = vifs[vifs > threshold]
            if candidates.empty:
                break

            if y is None:
                worst, worst_vif, reason = candidates.index[0], candidates.iloc[0], "highest VIF"
            else:
                corr_with_y = X[candidates.index].corrwith(y).abs()
                worst = corr_with_y.idxmin()
                worst_vif = candidates[worst]
                reason = f"lowest corr. with y ({corr_with_y[worst]:.3f})"

            remaining.remove(worst)
            dropped.append((worst, worst_vif, reason))
            if verbose:
                print(f"[VIF] Removing  {worst:<20} VIF={worst_vif:.2f}  ({reason})")

    return remaining, dropped


# ---------------------------------------------------------------------------
# Model evaluation criteria
# ---------------------------------------------------------------------------

# criterion -> (score_fn(model) -> float, is_better(new, old) -> bool)
CRITERIA = {
    "aic":    (lambda m: m.aic,          lambda new, old: new < old),
    "bic":    (lambda m: m.bic,          lambda new, old: new < old),
    "r2_adj": (lambda m: m.rsquared_adj, lambda new, old: new > old),
}


def _fit_ols(X: pd.DataFrame, y: pd.Series, features: list):
    """
    Fit an OLS model on the given subset of features (with an intercept).

    :param X: DataFrame with all available features
    :param y: Target Series
    :param features: list of feature names to include in the model
    :return: fitted statsmodels OLS results object, or None if features is empty
    """
    return sm.OLS(y, sm.add_constant(X[features])).fit() if features else None


# ---------------------------------------------------------------------------
# Stepwise selection
# ---------------------------------------------------------------------------

def stepwise_selection(
    X: pd.DataFrame,
    y: pd.Series,
    initial_features=None,
    criterion: str = "pvalue",
    threshold_in: float = 0.05,
    threshold_out: float = 0.10,
    verbose: bool = False,
):
    """
    Bidirectional stepwise selection (forward + backward at each iteration).

    :param X: DataFrame with the candidate features
    :param y: Target Series
    :param initial_features: optional list of features to start with
    :param criterion: "pvalue" (add lowest p-value < threshold_in, remove
        highest p-value > threshold_out) or "aic"/"bic"/"r2_adj" (add/remove
        whichever feature most improves that score; a feature is only
        added if it actually improves the score)
    :param threshold_in: p-value threshold to add a feature (pvalue criterion)
    :param threshold_out: p-value threshold to remove a feature (pvalue criterion)
    :param verbose: print each addition/removal
    :return: (selected_features, final_model)
    """
    included = list(initial_features) if initial_features else []
    all_features = list(X.columns)
    by_score = criterion != "pvalue"
    score_fn, is_better = CRITERIA.get(criterion, (None, None))

    current_score = score_fn(_fit_ols(X, y, included)) if by_score and included else None

    while True:
        changed = False
        excluded = [f for f in all_features if f not in included]

        # ---- forward step ----
        if not by_score:
            pvals = {f: _fit_ols(X, y, included + [f]).pvalues[f] for f in excluded}
            if pvals:
                best_feature = min(pvals, key=pvals.get)
                if pvals[best_feature] < threshold_in:
                    included.append(best_feature)
                    changed = True
                    if verbose:
                        print(f"Adding    {best_feature:<20} p={pvals[best_feature]:.5f}")
        else:
            best_feature, best_score = None, current_score
            for f in excluded:
                score = score_fn(_fit_ols(X, y, included + [f]))
                if best_score is None or is_better(score, best_score):
                    best_feature, best_score = f, score
            if best_feature is not None:
                included.append(best_feature)
                current_score = best_score
                changed = True
                if verbose:
                    print(f"Adding    {best_feature:<20} {criterion}={best_score:.4f}")

        # ---- backward step ----
        if included:
            if not by_score:
                model = _fit_ols(X, y, included)
                pvalues = model.pvalues.drop("const", errors="ignore")
                if not pvalues.empty and pvalues.max() > threshold_out:
                    worst_feature = pvalues.idxmax()
                    included.remove(worst_feature)
                    changed = True
                    if verbose:
                        print(f"Removing  {worst_feature:<20} p={pvalues.max():.5f}")
            else:
                worst_feature, worst_score = None, current_score
                for f in included:
                    model = _fit_ols(X, y, [x for x in included if x != f])
                    if model is None:
                        continue
                    score = score_fn(model)
                    if is_better(score, worst_score):
                        worst_feature, worst_score = f, score
                if worst_feature is not None:
                    included.remove(worst_feature)
                    current_score = worst_score
                    changed = True
                    if verbose:
                        print(f"Removing  {worst_feature:<20} {criterion}={worst_score:.4f}")

        if not changed:
            break

    final_model = _fit_ols(X, y, included)
    return included, final_model


# ---------------------------------------------------------------------------
# Runner for multiple targets
# ---------------------------------------------------------------------------

def _resolve_allowed_features(
    target: str,
    feature_cols: list,
    target_cols: list,
    exclude_features: dict,
    exclude_features_all: list,
    auto_exclude_other_targets: bool,
):
    """
    Build the list of features allowed as predictors for a given target.

    :param target: name of the current target column
    :param feature_cols: full candidate list of feature columns
    :param target_cols: list of all target columns being modeled
    :param exclude_features: dict {target: [features to exclude for
        that target only]}
    :param exclude_features_all: features to exclude from every model
    :param auto_exclude_other_targets: if True, remove the other target
        columns from the feature list (avoids information leakage)
    :return: (allowed_features, manually_excluded_features)
    """
    allowed = list(feature_cols)

    if auto_exclude_other_targets:
        other_targets = [t for t in target_cols if t != target]
        allowed = [f for f in allowed if f not in other_targets]

    manual_excl = list(exclude_features_all) + list(exclude_features.get(target, []))
    allowed = [f for f in allowed if f not in manual_excl and f != target]

    return allowed, manual_excl


def _auto_max_features(n_samples: int, n_features: int, max_features_ratio: float, target: str):
    """
    Decide an automatic feature cap when there are more (or as many)
    features as samples, printing a warning about it.

    :param n_samples: number of rows available for this target
    :param n_features: number of candidate features for this target
    :param max_features_ratio: fraction of n_samples to keep as the cap
    :param target: name of the target (used in the printed message)
    :return: the feature cap to use, or None if no cut is needed
    """
    if n_features < n_samples:
        return None

    cap = max(2, int(n_samples * max_features_ratio))
    print(
        f"[AUTO] Target '{target}': {n_features} features for {n_samples} samples. "
        f"Cutting to at most {cap} (max_features_ratio={max_features_ratio}) before "
        f"VIF/stepwise, keeping those most correlated with y. Set 'max_features_before_vif' "
        f"manually for a different value."
    )
    return cap


def run_for_multiple_targets(
    df: pd.DataFrame,
    feature_cols: list,
    target_cols: list,
    criterion: str = "pvalue",
    threshold_in: float = 0.15,
    threshold_out: float = 0.25,
    vif_threshold: float = 5.0,
    vif_target_aware: bool = True,
    corr_prescreen: float = 0.95,
    max_features_before_vif: int = None,
    max_features_ratio: float = 0.2,
    exclude_features: dict = None,
    exclude_features_all: list = None,
    auto_exclude_other_targets: bool = False,
    verbose: bool = False,
):
    """
    Run stepwise_selection for each target in target_cols, with an
    optional VIF collinearity filter beforehand.

    :param vif_target_aware: if True, VIF filtering uses this target's y
        to break ties (recommended); if False, uses global VIF only
    :param max_features_before_vif: manual cap on features entering VIF
        (needed when features >= samples); if None, derived automatically
        from max_features_ratio when features >= samples
    :param exclude_features: {target: [features excluded for that target]}
    :param exclude_features_all: features excluded from every model
    :param auto_exclude_other_targets: drop other target_cols from each
        target's feature set (avoids leakage)
    :return: {target: {selected_features, model, r2, r2_adj,
        dropped_by_vif, excluded_manually}}
    """
    results = {}
    exclude_features = exclude_features or {}
    exclude_features_all = exclude_features_all or []

    for target in target_cols:
        allowed, manual_excl = _resolve_allowed_features(
            target, feature_cols, target_cols,
            exclude_features, exclude_features_all,
            auto_exclude_other_targets,
        )

        if verbose:
            print(f"\n=== Target: {target} (criterion={criterion}) ===")
            if manual_excl:
                print(f"Manually excluded: {manual_excl}")

        sub = df[allowed + [target]].dropna()
        n_samples, n_features = sub.shape[0], len(allowed)

        effective_max_features = max_features_before_vif or _auto_max_features(
            n_samples, n_features, max_features_ratio, target,
        )

        if vif_threshold is None:
            features_ok, dropped_by_vif = list(allowed), []
        else:
            y_for_vif = sub[target] if vif_target_aware else None
            if verbose:
                print(f"[VIF] Target: {target}")
            features_ok, dropped_by_vif = filter_by_vif(
                sub[allowed], y=y_for_vif, threshold=vif_threshold,
                corr_prescreen=corr_prescreen,
                max_features_before_vif=effective_max_features,
                verbose=verbose,
            )

        selected, model = stepwise_selection(
            sub[features_ok], sub[target],
            criterion=criterion,
            threshold_in=threshold_in,
            threshold_out=threshold_out,
            verbose=verbose,
        )

        results[target] = {
            "selected_features": selected,
            "model": model,
            "r2": model.rsquared if model is not None else None,
            "r2_adj": model.rsquared_adj if model is not None else None,
            "dropped_by_vif": dropped_by_vif,
            "excluded_manually": manual_excl,
        }

    return results


# ---------------------------------------------------------------------------
# Human-readable summary of results
# ---------------------------------------------------------------------------

def summarize_results(results: dict, decimals: int = 4):
    """Print, per target, selected features, coefficients/p-values and R^2 of the final model."""
    for target, info in results.items():
        print(f"\n{'=' * 70}\nTarget: {target}\n{'=' * 70}")

        model = info["model"]
        if model is None or not info["selected_features"]:
            print("No feature was selected for this target.")
            continue

        print(f"Selected features: {info['selected_features']}")
        print(f"R^2          = {info['r2']:.{decimals}f}")
        print(f"R^2 adjusted = {info['r2_adj']:.{decimals}f}")

        if info["excluded_manually"]:
            print(f"Manually excluded: {info['excluded_manually']}")
        if info["dropped_by_vif"]:
            removed = [f for f, _, _ in info["dropped_by_vif"]]
            print(f"Removed for collinearity (VIF/corr): {removed}")

        coef_table = pd.DataFrame({
            "coef": model.params,
            "std_err": model.bse,
            "p_value": model.pvalues,
        }).round(decimals)
        print("\nFinal model coefficients:")
        print(coef_table.to_string())


def get_selected_features_dict(results: dict) -> dict:
    """{target: [selected features]} - simplified view of run_for_multiple_targets output."""
    return {target: info["selected_features"] for target, info in results.items()}


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """One row per target: target, n_features, features, r2, r2_adj - handy for Excel/CSV export."""
    rows = [{
        "target": target,
        "n_features": len(info["selected_features"]),
        "features": ", ".join(info["selected_features"]),
        "r2": info["r2"],
        "r2_adj": info["r2_adj"],
    } for target, info in results.items()]
    return pd.DataFrame(rows)