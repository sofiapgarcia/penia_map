import matplotlib.pyplot as plt
import pandas as pd

def plot_distribution(df, column, bins=10):
    """
    Plot the distribution of a numerical column using a histogram.

    :param df: Input DataFrame
    :param column: Name of the column to visualize
    :param bins: Number of bins for the histogram
    :return: None (displays the plot)
    """

    # Check if the column exists in the DataFrame
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return

    # Convert column to numeric (invalid values become NaN)
    data = pd.to_numeric(df[column], errors="coerce")

    # Check if the column has valid numeric data
    if data.isna().all():
        print(f"Column '{column}' has no valid numeric values.")
        return

    # Plot histogram
    data.plot(kind="hist", bins=bins)

    # Set plot title and labels
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    # Display the plot
    plt.show()