import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_distribution(df, column, bins="auto", bin_width=None):
    """
    Plot the distribution of a numerical column using a histogram.

    :param df: Input DataFrame
    :param column: Name of the column to visualize
    :param bins:
        - "auto" -> automatic bins (default)
        - int -> number of bins
        - "step1" -> bin size = 1
    :param bin_width: If set, overrides bins and uses fixed bin width
    :return: None (displays plot)
    """

    # Check column exists
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return

    # Clean data
    data = pd.to_numeric(df[column], errors="coerce").dropna()

    if data.empty:
        print(f"Column '{column}' has no valid numeric values.")
        return

    # -----------------------------
    # PRIORITY 1: bin width mode
    # -----------------------------
    if bin_width is not None:
        bin_edges = np.arange(data.min(), data.max() + bin_width, bin_width)

    # -----------------------------
    # PRIORITY 2: predefined modes
    # -----------------------------
    elif bins == "step1":
        bin_edges = np.arange(data.min(), data.max() + 1, 1)

    elif isinstance(bins, int):
        bin_edges = bins

    else:
        bin_edges = "auto"

    # Plot
    plt.hist(data, bins=bin_edges, edgecolor="black")

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.show()