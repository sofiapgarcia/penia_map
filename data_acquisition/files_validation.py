import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np

def extract_data_from_file(file_path):
    """
    Extract metadata and signal data from an OpenSignals text file.

    :param file_path: Path to the OpenSignals .txt file
    :return: Tuple containing:
             - params (dict): sampling rate, date and time metadata
             - df (DataFrame): signal data with columns such as nSeq and channels
    """
    metadata = None
    data_start = 0

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith("# {"):
            metadata = json.loads(line[2:])
        if "EndOfHeader" in line:
            data_start = i + 1
            break

    device_info = list(metadata.values())[0]
    df = pd.read_csv(
        file_path,
        sep="\t",
        skiprows=data_start,
        header=None,
        usecols=range(len(device_info["column"]))
    )
    df.columns = device_info["column"]

    emg_indices = [i for i, col in enumerate(df.columns) if col.upper() == "EMG"]

    if len(emg_indices) >= 2:
        # 1º = extensor, 2º = flexor
        df.columns.values[emg_indices[0]] = "EMG_Extensor"
        df.columns.values[emg_indices[1]] = "EMG_Flexor"

    filename = os.path.basename(file_path)
    _, condition = parse_filename(filename)

    if condition != "HD":
        df.columns.values[-3:] = ["x", "y", "z"]

    params = {
        "sampling rate": device_info["sampling rate"],
        "date": device_info["date"],
        "time": device_info["time"]
    }

    return params, df


def parse_filename(file):
    """
    Parse subject ID and condition from filename.

    :param file: Filename in format 'Sx_HI.txt', 'Sx_HD.txt' or 'Sx_MVC_FLEXOR.txt'
    :return: Tuple containing:
             - subject (str): subject ID (e.g., S1)
             - condition (str): experimental test (HI, HD or MVC_FLEXOR)
    """
    name = file.replace(".txt", "")

    subject_raw, condition = name.split("_", 1)

    subject = subject_raw.replace("S", "")

    return subject, condition

def analyze_sequence(sequence):
    """
    Analyze nSeq sequence integrity by detecting wraps, missing samples,
    invalid jumps, and computing signal loss percentage.

    :param sequence: pandas Series containing nSeq values from OpenSignals data
    :return: Dictionary with:
             - wraps: number of detected counter wraps
             - missing_samples: estimated number of missing samples due to gaps
             - expected_samples: total expected samples in the sequence
             - loss_percentage: percentage of missing samples
             - invalid_jumps: Series with abnormal jumps in nSeq
    """

    # Compute difference between consecutive samples
    diff = sequence.diff()

    # Detect valid counter wraps
    wraps = (diff == -32767).sum()

    # Estimate missing samples from sequence gaps
    missing_samples = (diff[diff > 1] - 1).sum()

    # Estimate total expected number of samples
    expected_samples = wraps * 32768 + sequence.iloc[-1] + 1

    # Compute percentage of missing samples
    loss_percentage = (
        missing_samples / expected_samples * 100
        if expected_samples > 0 else 0
    )

    invalid_file = loss_percentage > 10

    # Identify invalid jumps
    invalid_jumps = diff[
        (diff != 1) &
        (diff != -32767) &
        (~diff.isna())
    ]

    return {
        "wraps": int(wraps),
        "missing_samples": int(missing_samples),
        "expected_samples": int(expected_samples),
        "acquired_samples": int(len(sequence)),
        "loss_percentage": round(loss_percentage, 2),
        "invalid_jumps": invalid_jumps,
        "invalid_file": invalid_file
    }


def plot_sensors_over_time(params, df, title=None):
    """
    Plot all sensor channels over acquisition time.

    :param params: Dictionary containing acquisition parameters (e.g., sampling rate)
    :param df: DataFrame containing sensor signals including nSeq and channels
    :param title: Optional title for the plot
    :return: Matplotlib figure object with all sensor signals plotted over time
    """

    # Extract sampling rate from acquisition parameters
    sampling_rate = params["sampling rate"]

    # Number of samples in the dataset
    n_samples = len(df)

    # Create time vector in seconds based on sampling rate
    time_seconds = np.arange(n_samples) / sampling_rate

    # Select all sensor columns except sequence counter
    sensor_cols = [col for col in df.columns if col != 'nSeq']

    # Number of sensors to plot
    n_sensors = len(sensor_cols)

    # Create subplot structure (one row per sensor)
    fig, axes = plt.subplots(
        n_sensors,
        1,
        figsize=(14, 3 * n_sensors),
        sharex=True
    )

    # Ensure axes is iterable even when only one sensor exists
    if n_sensors == 1:
        axes = [axes]

    # Plot each sensor signal over time
    for idx, (ax, sensor) in enumerate(zip(axes, sensor_cols)):

        ax.plot(
            time_seconds,
            df[sensor],
            linewidth=1.5,
            alpha=0.8,
            color=f"C{idx}"
        )

        # Label each subplot with sensor name
        ax.set_ylabel(sensor)

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

    # Label x-axis only on last subplot
    axes[-1].set_xlabel("Time (s)")

    # Add global title if provided
    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    return fig

if __name__ == '__main__':

    base_path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos"

    selected_subject = "S9"

    subject_path = os.path.join(base_path, selected_subject)

    if not os.path.isdir(subject_path):
        print(f"Subject {selected_subject} not found.")
        exit()

    print(f"SUBJECT: {selected_subject}")

    results = []

    for file in os.listdir(subject_path):

        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(subject_path, file)

        try:
            params, df = extract_data_from_file(file_path)

            sequence = df["nSeq"]

            analysis = analyze_sequence(sequence)

            print(f"\nFile: {file}")
            print(f"Loss percentage: {analysis['loss_percentage']:.2f}%")

            results.append({
                "file": file,
                "subject": selected_subject,
                "wraps": analysis["wraps"],
                "missing_samples": analysis["missing_samples"],
                "expected_samples": analysis["expected_samples"],
                "acquired_samples": analysis["acquired_samples"],
                "loss_percentage": analysis["loss_percentage"],
                "invalid_jumps": len(analysis["invalid_jumps"]),
                "invalid_file": analysis["invalid_file"]
            })

            invalid_jumps = analysis["invalid_jumps"]
            for idx, value in invalid_jumps.items():
                prev_value = sequence.iloc[idx - 1]
                current_value = sequence.iloc[idx]

                print(f"{idx}: {prev_value} -> {current_value} (diff={value})")

            plot_sensors_over_time(params, df, title=f"{selected_subject} - {file}")
            plt.show()

        except Exception as e:
            print(f"Error in {file}: {e}")

    df_results = pd.DataFrame(results)

    output_path = os.path.join(subject_path, f"{selected_subject}_files_validation.xlsx")

    df_results.to_excel(output_path, index=False)

    print("\nExcel saved at:")
    print(output_path)