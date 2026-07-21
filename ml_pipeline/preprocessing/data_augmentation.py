"""
Functions to perform data augmentation on tabular datasets using a combination
of SMOGN (Synthetic Minority Over-sampling Technique for Regression with Gaussian Noise)
and Gaussian Copula-based synthetic data generation.

This module provides utilities to handle imbalanced regression problems by:
- Oversampling rare target regions using SMOGN
- Generating additional synthetic samples using a probabilistic model (Gaussian Copula)

Available Functions
-------------------
[Public]
apply_data_augmentation(...): Apply a full data augmentation pipeline combining SMOGN and Gaussian Copula.

[Private]
_apply_smogn(...): Perform SMOGN-based resampling to balance the target distribution.
_augment_dataframe_gaussian_copula(...): Generate synthetic tabular data using a Gaussian Copula model.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import smogn
from packaging.metadata import Metadata
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def apply_data_augmentation(
    df,
    k=2,
    target_column="Estimated metabolic age",
    rel_ctrl_pts_rg=None,
    samp_method="extreme",
    rel_method="manual",
    preserve_original=True,
    n_samples: int = 20,
    verbose=True
):
    """
    Apply SMOGN followed by Gaussian Copula augmentation to the dataset.

    :param df: dataset (pandas DataFrame)
    :param k: Number of neighbours to use in SMOGN
    :param target_column: Target column name
    :param rel_ctrl_pts_rg: Relevance control points for SMOGN
    :param samp_method: Sampling method ("extreme", "balance", etc.)
    :param rel_method: Relevance method ("manual" or "auto")
    :param preserve_original: If True, re-add original data after SMOGN
    :param n_samples: Number of synthetic rows to generate with Gaussian Copula.
                      If None, uses the size of the SMOGN-augmented dataset
    :param verbose: If True, print dataset sizes
    :return: Fully augmented dataset (SMOGN + Gaussian Copula)
    """

    # Step 1: Apply SMOGN
    """
    df = _apply_smogn(df=df, k=k, target_column=target_column, rel_ctrl_pts_rg=rel_ctrl_pts_rg,
                      samp_method=samp_method, rel_method=rel_method, preserve_original=preserve_original,
                      verbose=False)
    """

    # Print intermediate dataset size
    if verbose:
        print("Sample size after SMOGN step:", len(df))

    # Step 2: Apply Gaussian Copula
    final_df = _augment_dataframe_gaussian_copula(df=df, n_samples=n_samples)

    # Print final dataset size
    if verbose:
        print("Sample size after full augmentation (SMOGN + Gaussian Copula):", len(final_df))

    return final_df



# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _apply_smogn(
    df,
    k=2,
    target_column="Estimated metabolic age",
    rel_ctrl_pts_rg=None,
    samp_method="extreme",
    rel_method="manual",
    preserve_original=True,
    verbose=False
):
    """
    Apply SMOGN augmentation to the training data.

    :param df: dataset (pandas DataFrame)
    :param k: Number of neighbours to use
    :param target_column: Target column name
    :param rel_ctrl_pts_rg: Relevance control points
    :param samp_method: Sampling method ("extreme", "balance", etc.)
    :param rel_method: Relevance method ("manual" or "auto")
    :param preserve_original: If True, re-add original data to avoid loss
    :param verbose: If True, print dataset sizes
    :return: Augmented dataset
    """

    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in DataFrame")

    data = df.copy()

    # Print initial dataset size
    if verbose:
        print("Sample size before data augmentation:", len(data))


    if rel_ctrl_pts_rg is None:
        rel_ctrl_pts_rg = [
            [49, 1, 0],
            [50, 1, 0],
            [52, 0, 0],
            [55, 0, 0],
            [60, 0, 0],
            [61, 1, 0],
            [63, 1, 0],
            [70, 0, 0]
        ]

    # Apply SMOGN
    # Performs oversampling of rare regions and possible undersampling of common ones
    augmented = smogn.smoter(
        data=data,
        y=target_column,
        k=k,
        samp_method=samp_method,
        rel_method=rel_method,
        rel_ctrl_pts_rg=rel_ctrl_pts_rg
    )

    # Preserve original data (optional)
    # Ensures no original samples are lost after SMOGN
    if preserve_original:
        combined = pd.concat([data, augmented])
        final_df = combined.drop_duplicates()

    else:
        final_df = augmented.reset_index(drop=True)

    # Print final dataset sizes
    if verbose:
        print("Sample size after SMOGN:", len(final_df))
    return final_df


def _augment_dataframe_gaussian_copula(df, n_samples: int = 20):
    """
    Augment a dataset using a Gaussian Copula synthetic data generator.

    :param df: Input DataFrame
    :param n_samples: Number of synthetic rows to generate. If None, uses the same size as the input DataFrame
    :return: DataFrame containing both original and synthetic data
    """

    if n_samples is None:
        n_samples = len(df)

    # Create metadata from the input dataframe (SDV 1.x compatible)
    metadata = Metadata.detect_from_dataframe(data=df)

    # Initialize and train synthesizer
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df)

    # Generate synthetic data
    synthetic_df = synthesizer.sample(num_rows=n_samples)

    # Combine original + synthetic data
    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)

    return augmented_df