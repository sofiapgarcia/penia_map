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

    Rule of thumb:
        VIF < 5   -> ok
        VIF 5-10  -> moderate collinearity, pay attention
        VIF > 10  -> strong collinearity, problematic
    """
    # statsmodels needs a constant column to compute VIF correctly
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
    Quickly prune collinear features using only the correlation matrix
    (no OLS model is fitted). This solves two problems that plain VIF
    cannot handle well when there are many features:

    1) Nearly duplicated / exactly collinear features (VIF would be
       infinite) - a simple pairwise correlation is enough here, no need
       to fit a model.
    2) Datasets with more features than samples (common when there are
       dozens of variables per muscle/frequency band/repetition and few
       participants). In this regime VIF is not even well defined (the
       R^2 of the auxiliary regression is exactly 1.0), so features are
       first reduced by correlation, and only afterwards (with a number
       of features already smaller than the number of samples) does it
       make sense to compute VIF properly.

    :param X: DataFrame with the features
    :param y: Target Series (optional). Used to decide which feature of a
        correlated pair to keep (the one more correlated with y)
    :param corr_threshold: absolute correlation above which a pair is
        considered redundant
    :param max_features: optional cap on the number of features to keep
    :param verbose: print each removal
    :return: tuple (kept_features, dropped) where dropped is a list of
        (feature, corr_value, reason) tuples

    In each iteration: find the pair of features with the highest
    absolute correlation between them; if it exceeds corr_threshold,
    remove one of them (the one with lower absolute correlation with y,
    if y is given; otherwise remove the second feature of the pair,
    arbitrarily). Repeat until no pair is above the threshold, or until
    max_features is reached (if specified).
    """
    remaining = list(X.columns)
    dropped = []

    while len(remaining) > 1:
        if max_features is not None and len(remaining) <= max_features:
            break

        corr = X[remaining].corr().abs().values.copy()
        n = len(remaining)
        # ignore the diagonal (self-correlation)
        idx = np.arange(n)
        corr[idx, idx] = 0.0

        max_val = corr.max()
        if max_val <= corr_threshold and (max_features is None or len(remaining) <= max_features):
            break
        if max_val <= corr_threshold and max_features is not None:
            # no more strongly correlated pairs, but still above max_features:
            # cut the feature least related to y
            if y is not None:
                corr_with_y = X[remaining].corrwith(y).abs()
                worst_feature = corr_with_y.idxmin()
                reason = f"cut for max_features (corr. with y={corr_with_y[worst_feature]:.3f})"
            else:
                worst_feature = remaining[-1]
                reason = "cut for max_features (arbitrary, no y given)"
            remaining.remove(worst_feature)
            dropped.append((worst_feature, None, reason))
            if verbose:
                print(f"[CORR] Removing {worst_feature:<20} {reason}")
            continue

        # locate the most correlated pair of features
        i, j = np.unravel_index(np.argmax(corr), corr.shape)
        f1, f2 = remaining[i], remaining[j]

        if y is not None:
            c1 = abs(X[f1].corr(y))
            c2 = abs(X[f2].corr(y))
            drop_feature = f1 if c1 < c2 else f2
        else:
            drop_feature = f2

        remaining.remove(drop_feature)
        dropped.append((drop_feature, max_val, f"corr. {max_val:.3f} with {f1 if drop_feature == f2 else f2}"))
        if verbose:
            print(f"[CORR] Removing {drop_feature:<20} corr={max_val:.3f}")

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
    Iteratively remove collinear features until every VIF is below the
    given threshold.

    :param X: DataFrame with the features
    :param y: Target Series (optional). If given, activates the
        target-aware mode
    :param threshold: maximum acceptable VIF
    :param corr_prescreen: correlation threshold used in the fast
        Phase 1 pruning. Use None to disable this phase (not
        recommended if you have many features)
    :param max_features_before_vif: max number of features to keep
        before entering Phase 2 (VIF via OLS). Strongly recommended to
        set this to a value smaller than the number of samples for wide
        datasets (many features, few rows) - otherwise VIF is not well
        defined. Rule of thumb: at most (n_samples / 10) to
        (n_samples / 5)
    :param verbose: print each removal
    :return: tuple (kept_features, dropped) where dropped is a list of
        (feature, vif_or_corr, reason) tuples, in removal order,
        covering both phases

    Runs in TWO phases:

    Phase 1 (fast, no OLS): `_prune_by_correlation`, which removes
    near-duplicate pairs (corr > corr_prescreen) and, if
    `max_features_before_vif` is given, cuts down to that number of
    features. This is essential when there are far more features than
    samples (otherwise VIF via OLS never converges and can take forever
    / give VIF=inf indefinitely).

    Phase 2: the "real" VIF filter (via statsmodels), already on a
    smaller, well-conditioned set of features.

    There are two modes for the tie-breaking criterion of who is
    removed:

    1) y=None ("global" mode, target-agnostic)
       Always removes the feature with the highest VIF.

    2) y given (target-aware mode, RECOMMENDED)
       At each iteration, looks at every feature whose VIF is above the
       threshold and removes the one with the LOWEST absolute
       correlation with y.
    """
    # --- Phase 1: fast pruning by correlation (no OLS) ---
    remaining = list(X.columns)
    dropped = []

    if corr_prescreen is not None or max_features_before_vif is not None:
        remaining, dropped_corr = _prune_by_correlation(
            X[remaining], y=y,
            corr_threshold=corr_prescreen if corr_prescreen is not None else 0.999999,
            max_features=max_features_before_vif,
            verbose=verbose,
        )
        dropped.extend(dropped_corr)

    # --- Phase 2: "real" VIF via OLS, now on a smaller set ---
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        while len(remaining) > 1:
            vifs = compute_vif(X[remaining])
            candidates = vifs[vifs > threshold]

            if candidates.empty:
                break

            if y is None:
                worst_feature = candidates.index[0]
                worst_vif = candidates.iloc[0]
                reason = "highest VIF"
            else:
                corr_with_y = X[candidates.index].corrwith(y).abs()
                worst_feature = corr_with_y.idxmin()
                worst_vif = candidates[worst_feature]
                reason = f"lowest corr. with y ({corr_with_y[worst_feature]:.3f})"

            remaining.remove(worst_feature)
            dropped.append((worst_feature, worst_vif, reason))
            if verbose:
                print(f"[VIF] Removing  {worst_feature:<20} VIF={worst_vif:.2f}  ({reason})")

    return remaining, dropped


# ---------------------------------------------------------------------------
# Model evaluation criteria
# ---------------------------------------------------------------------------

def _score(model, criterion: str) -> float:
    """
    Return the value of the given criterion for a fitted model.

    :param model: fitted statsmodels OLS results object
    :param criterion: one of "aic", "bic", "r2_adj"
    :return: the criterion's value for this model
    """
    if criterion == "aic":
        return model.aic
    if criterion == "bic":
        return model.bic
    if criterion == "r2_adj":
        return model.rsquared_adj
    raise ValueError(f"Unknown criterion for _score: {criterion}")


def _better(new_score: float, old_score: float, criterion: str) -> bool:
    """
    Check whether new_score is better than old_score for the given criterion.

    :param new_score: candidate score
    :param old_score: reference score to compare against
    :param criterion: one of "aic", "bic", "r2_adj"
    :return: True if new_score improves on old_score
    """
    if criterion in ("aic", "bic"):
        return new_score < old_score   # we want to minimize
    if criterion == "r2_adj":
        return new_score > old_score   # we want to maximize
    raise ValueError(f"Unknown criterion for _better: {criterion}")


def _fit_ols(X: pd.DataFrame, y: pd.Series, features: list):
    """
    Fit an OLS model on the given subset of features (with an intercept).

    :param X: DataFrame with all available features
    :param y: Target Series
    :param features: list of feature names to include in the model
    :return: fitted statsmodels OLS results object, or None if features is empty
    """
    if not features:
        return None
    return sm.OLS(y, sm.add_constant(X[features])).fit()


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
    :param criterion: selection criterion, one of:
        - "pvalue": add the feature with the lowest p-value
          (< threshold_in); remove the feature with the highest p-value
          (> threshold_out).
        - "aic": add/remove the feature that most reduces the model's AIC.
        - "bic": add/remove the feature that most reduces the model's BIC.
        - "r2_adj": add/remove the feature that most increases adjusted R^2.
    :param threshold_in: p-value threshold to add a feature (pvalue criterion)
    :param threshold_out: p-value threshold to remove a feature (pvalue criterion)
    :param verbose: print each addition/removal
    :return: tuple (selected_features, model) where selected_features is
        the list of chosen features (in entry order) and model is the
        final fitted statsmodels OLS results object

    For "aic"/"bic"/"r2_adj", a feature is only added if it actually
    improves the model's score (otherwise the process simply stops).
    """
    included = list(initial_features) if initial_features else []
    all_features = list(X.columns)

    current_score = None
    if criterion in ("aic", "bic", "r2_adj") and included:
        current_score = _score(_fit_ols(X, y, included), criterion)

    while True:
        changed = False

        # ---- FORWARD STEP ----
        excluded = [f for f in all_features if f not in included]

        if criterion == "pvalue":
            new_pval = pd.Series(index=excluded, dtype=float)
            for f in excluded:
                model = _fit_ols(X, y, included + [f])
                new_pval[f] = model.pvalues[f]

            if not new_pval.empty:
                best_pval = new_pval.min()
                if best_pval < threshold_in:
                    best_feature = new_pval.idxmin()
                    included.append(best_feature)
                    changed = True
                    if verbose:
                        print(f"Adding    {best_feature:<20} p={best_pval:.5f}")

        else:  # aic / bic / r2_adj
            best_feature, best_score = None, current_score
            for f in excluded:
                model = _fit_ols(X, y, included + [f])
                score = _score(model, criterion)
                if best_score is None or _better(score, best_score, criterion):
                    best_score, best_feature = score, f

            if best_feature is not None:
                included.append(best_feature)
                current_score = best_score
                changed = True
                if verbose:
                    print(f"Adding    {best_feature:<20} {criterion}={best_score:.4f}")

        # ---- BACKWARD STEP ----
        if included:
            if criterion == "pvalue":
                model = _fit_ols(X, y, included)
                pvalues = model.pvalues.drop("const", errors="ignore")
                worst_pval = pvalues.max() if not pvalues.empty else 0
                if worst_pval > threshold_out:
                    worst_feature = pvalues.idxmax()
                    included.remove(worst_feature)
                    changed = True
                    if verbose:
                        print(f"Removing  {worst_feature:<20} p={worst_pval:.5f}")

            else:  # aic / bic / r2_adj
                worst_feature, worst_score = None, current_score
                for f in included:
                    remaining_features = [x for x in included if x != f]
                    model = _fit_ols(X, y, remaining_features)
                    score = _score(model, criterion) if model is not None else None
                    if model is not None and _better(score, worst_score, criterion):
                        worst_score, worst_feature = score, f

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
    :return: tuple (allowed_features, manually_excluded_features)
    """
    allowed = list(feature_cols)

    if auto_exclude_other_targets:
        other_targets = [t for t in target_cols if t != target]
        allowed = [f for f in allowed if f not in other_targets]

    manual_excl = list(exclude_features_all) + list(exclude_features.get(target, []))
    allowed = [f for f in allowed if f not in manual_excl]

    # the target itself can never be its own feature
    allowed = [f for f in allowed if f != target]

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

    # more features than samples: regression is not even well defined,
    # so automatically reduce to a safe fraction of the sample size
    cap = max(2, int(n_samples * max_features_ratio))
    print(
        f"[AUTO] Target '{target}': {n_features} features for {n_samples} "
        f"samples. Automatically cutting down to at most {cap} features "
        f"(max_features_ratio={max_features_ratio}) before VIF/stepwise, "
        f"choosing the ones most correlated with this y. Adjust "
        f"'max_features_ratio' or set 'max_features_before_vif' manually "
        f"if you want a different value."
    )
    return cap


def run_for_multiple_targets(
    df: pd.DataFrame,
    feature_cols: list,
    target_cols: list,
    criterion: str = "pvalue",
    threshold_in: float = 0.05,
    threshold_out: float = 0.10,
    vif_threshold: float = 5.0,
    vif_target_aware: bool = True,
    corr_prescreen: float = 0.95,
    max_features_before_vif: int = None,
    max_features_ratio: float = 0.2,
    exclude_features: dict = None,
    exclude_features_all: list = None,
    auto_exclude_other_targets: bool = True,
    verbose: bool = False,
):
    """
    Run stepwise_selection for several targets at once, with an optional
    collinearity (VIF) filter applied before the stepwise search.

    :param df: source DataFrame containing features and targets
    :param feature_cols: candidate feature columns
    :param target_cols: target columns to model, one at a time
    :param criterion: stepwise selection criterion (see stepwise_selection)
    :param threshold_in: p-value threshold to add a feature
    :param threshold_out: p-value threshold to remove a feature
    :param vif_threshold: max acceptable VIF. None disables the filter
        (uses every feature)
    :param vif_target_aware: if True (recommended), the VIF filter runs
        separately FOR EACH target: when two features are collinear, it
        keeps the one more correlated with that specific y
    :param corr_prescreen: correlation threshold (0-1) for a fast
        pre-filter (no OLS) applied before the real VIF filter. Removes
        near-duplicate pairs. Greatly reduces VIF's computational cost
        when there are many features. None disables it
    :param max_features_before_vif: if you have MANY features (e.g.
        dozens of EMG variables) relative to few samples, set here a
        maximum number of features to keep BEFORE entering VIF via OLS
        (which is only well defined when n_features < n_samples). Rule
        of thumb: something between n_samples/10 and n_samples/3. If
        None and the dataset has more (or as many) features as samples,
        the cut is done AUTOMATICALLY using max_features_ratio
    :param max_features_ratio: used only when max_features_before_vif is
        None and there are more (or as many) features as samples.
        Automatically cuts down to max(2, int(n_samples * ratio))
        features before VIF, choosing by correlation with y. Default
        0.2 (e.g. 40 samples -> max. 8 features)
    :param exclude_features: optional dict {target: [features to
        exclude only for that target]}. Use this when you know, for
        domain reasons (e.g. information leakage, variables derived
        from the target itself, etc.), that certain features cannot
        enter the model for a specific target.
        E.g.: {"Age (y)": ["Estimated metabolic age"]}
    :param exclude_features_all: optional list of features to exclude
        from ALL models, regardless of target
    :param auto_exclude_other_targets: if True (default), automatically
        removes the remaining target_cols from each target's feature
        set. This avoids the common mistake of using
        `feature_cols=list(df.columns)` and ending up with one target
        being used as a predictor of another (information leakage /
        artificial multicollinearity)
    :param verbose: print details of each step
    :return: dict in the format:
        {
            "y1": {
                "selected_features": [...],
                "model": <OLS results>,
                "r2_adj": 0.83,
                "dropped_by_vif": [("f2", 12.3, "lowest corr. with y (0.10)"), ...],
                "excluded_manually": ["Estimated metabolic age"],
            },
            ...
        }
    """
    results = {}
    exclude_features = exclude_features or {}
    exclude_features_all = exclude_features_all or []

    for target in target_cols:
        # --- build the list of features allowed for THIS target ---
        allowed, manual_excl = _resolve_allowed_features(
            target, feature_cols, target_cols,
            exclude_features, exclude_features_all,
            auto_exclude_other_targets,
        )

        if verbose:
            print(f"\n=== Target: {target} (criterion={criterion}) ===")
            if manual_excl:
                print(f"Manually excluded: {manual_excl}")

        cols = allowed + [target]
        sub = df[cols].dropna()

        n_samples, n_features = sub.shape[0], len(allowed)

        # decide the feature cap before VIF (manual or automatic)
        effective_max_features = max_features_before_vif
        if effective_max_features is None:
            effective_max_features = _auto_max_features(
                n_samples, n_features, max_features_ratio, target,
            )

        if vif_threshold is None:
            features_ok, dropped_by_vif = list(allowed), []
        else:
            # target-aware mode uses this target's y; global mode uses y=None
            y_for_vif = sub[target] if vif_target_aware else None
            if verbose:
                print(f"[VIF] Target: {target}")
            features_ok, dropped_by_vif = filter_by_vif(
                sub[allowed], y=y_for_vif, threshold=vif_threshold,
                corr_prescreen=corr_prescreen,
                max_features_before_vif=effective_max_features,
                verbose=verbose,
            )

        X = sub[features_ok]
        y = sub[target]

        selected, model = stepwise_selection(
            X, y,
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
    """
    Print, for each target, the chosen features, coefficients, p-values
    and R^2 (and adjusted R^2) of the final model.

    :param results: the dict returned by run_for_multiple_targets
    :param decimals: number of decimal places to display
    :return: None (prints to stdout)
    """
    for target, info in results.items():
        print(f"\n{'=' * 70}")
        print(f"Target: {target}")
        print(f"{'=' * 70}")

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

        # coefficient / p-value table
        coef_table = pd.DataFrame({
            "coef": model.params,
            "std_err": model.bse,
            "p_value": model.pvalues,
        }).round(decimals)
        print("\nFinal model coefficients:")
        print(coef_table.to_string())


def get_selected_features_dict(results: dict) -> dict:
    """
    Simplify the output of run_for_multiple_targets to the essentials:
    a dict {target: [features of the best model]}.

    :param results: the dict returned by run_for_multiple_targets
    :return: dict mapping each target to its list of selected features

    Example:
        {"Age (y)": ["f1", "f3"], "peak_power": ["f2", "f5"]}
    """
    return {target: info["selected_features"] for target, info in results.items()}


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """
    Tabular alternative: return a DataFrame with one row per target,
    useful for exporting to Excel/CSV or pasting into a report.

    :param results: the dict returned by run_for_multiple_targets
    :return: DataFrame with columns: target, n_features, features, r2, r2_adj
    """
    rows = []
    for target, info in results.items():
        rows.append({
            "target": target,
            "n_features": len(info["selected_features"]),
            "features": ", ".join(info["selected_features"]),
            "r2": info["r2"],
            "r2_adj": info["r2_adj"],
        })
    return pd.DataFrame(rows)