import pandas as pd
import os
import matplotlib.pyplot as plt

# internal imports
from data_acquisition.data_loader import load_subject_data, analyze_sequence
from data_acquisition.data_visualizer import plot_sensors_over_time

if __name__ == '__main__':

    base_path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos"

    selected_subject = "S32"

    subject_path = os.path.join(base_path, selected_subject)

    if not os.path.isdir(subject_path):
        print(f"Subject {selected_subject} not found.")
        exit()

    print(f"SUBJECT: {selected_subject}")

    results = []

    subject_data = load_subject_data(base_path, selected_subject)

    non_condition_keys = {"potentiometer_position", "HI_values"}

    for condition, content in subject_data.items():

        if condition in non_condition_keys:
            continue

        params = content["params"]
        df = content["data"]

        try:
            sequence = df["nSeq"]

            analysis = analyze_sequence(sequence)

            print(f"\nCondition: {condition}")
            print(f"Loss percentage: {analysis['loss_percentage']:.2f}%")

            results.append({
                "file": condition,
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

            plot_sensors_over_time(params, df, title=f"{selected_subject} - {condition}")
            plt.show()

        except Exception as e:
            print(f"Error in {condition}: {e}")

    df_results = pd.DataFrame(results)

    output_path = os.path.join(subject_path, f"{selected_subject}_files_validation.xlsx")

    df_results.to_excel(output_path, index=False)

    print("\nExcel saved at:")
    print(output_path)