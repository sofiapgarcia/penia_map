import pandas as pd
import os
import matplotlib.pyplot as plt

# internal imports
from data_acquisition.data_loader import load_subject_data, analyze_sequence
from data_acquisition.data_visualizer import plot_sensors_over_time

subjects_with_no_calibration = ["S4","S14","S17","S22","S24","S25","S84"]

if __name__ == '__main__':

    base_path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos"

    # Get all subject folders
    subjects = [
        folder for folder in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, folder))
    ]
    # if one specific subject
    #subjects = ["S46"]

    print(f"Found {len(subjects)} subjects.")

    results = []

    non_condition_keys = {
        "handgrip_position",
        "handgrip_position_source",
        "HI_values",
        "calibration"
    }

    for subject in sorted(subjects):

        if subject in subjects_with_no_calibration:
            continue

        print(f"\n{'=' * 60}")
        print(f"SUBJECT: {subject}")
        print(f"{'=' * 60}")

        try:
            subject_data = load_subject_data(base_path, subject)

            for condition, content in subject_data.items():

                if condition in non_condition_keys:
                    continue

                params = content["params"]
                df = content["data"]

                try:
                    sequence = df["nSeq"]

                    analysis = analyze_sequence(sequence)

                    print(f"\nCondition: {condition}")
                    print(
                        f"Loss percentage: "
                        f"{analysis['loss_percentage']:.2f}%"
                    )

                    results.append({
                        "file": condition,
                        "subject": subject,

                        "wraps": analysis["wraps"],
                        "missing_samples": analysis["missing_samples"],
                        "expected_samples": analysis["expected_samples"],
                        "acquired_samples": analysis["acquired_samples"],
                        "loss_percentage": analysis["loss_percentage"],

                        # Gap continuity
                        "number_of_gaps": analysis["number_of_gaps"],
                        "max_consecutive_missing": analysis["max_consecutive_missing"],
                        "mean_gap_length": analysis["mean_gap_length"],
                        "median_gap_length": analysis["median_gap_length"],
                        "longest_gap_duration_s": analysis["longest_gap_duration_s"],
                        "max_gap_percentage": analysis["max_gap_percentage"],

                        "invalid_jumps": len(analysis["invalid_jumps"]),
                        "invalid_file": analysis["invalid_file"]
                    })

                    invalid_jumps = analysis["invalid_jumps"]

                    for idx, value in invalid_jumps.items():
                        prev_value = sequence.iloc[idx - 1]
                        current_value = sequence.iloc[idx]

                        print(
                            f"{idx}: {prev_value} -> {current_value} "
                            f"(diff={value})"
                        )

                    # Plot each condition
                    plot_sensors_over_time(params,df,title=f"{subject} - {condition}")
                    plt.show()

                except Exception as e:
                    print(f"Error in {condition}: {e}")

        except Exception as e:
            print(f"Error loading subject {subject}: {e}")

    # Create dataframe with all results
    df_results = pd.DataFrame(results)

    # Save one Excel containing all subjects
    output_path = os.path.join(
        base_path,
        "files_validation.xlsx"
    )

    df_results.to_excel(output_path, index=False)

    print("\nExcel saved at:")
    print(output_path)