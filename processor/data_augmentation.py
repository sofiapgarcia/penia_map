"""
Functions to...

Available Functions
-------------------
[Public]
function1(...):

[Private]
_function2(...):
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

def apply_smogn(
    df,
    k=4,
    target_column="Estimated metabolic age",
    rel_ctrl_pts_rg=None,
    samp_method="balance",
    rel_method="manual",
    preserve_original=True,
    verbose=True
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
            [55, 0, 0],
            [60, 0, 0],
            [64, 1, 0],
            [71, 0, 0]
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


def augment_dataframe_gaussian_copula(df, n_samples: int = None):
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

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #