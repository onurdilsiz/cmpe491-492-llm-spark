import os
import json
import csv

def create_csv_from_jsons(json_dir, output_csv):
    # Prepare data for CSV
    csv_data = []
    cases_map = {
        "detected0": ["RDD vs DataFrame","occurrences0"],
        "detected1": ["Coalesce vs Repartition","occurrences1"],
        "detected2": ["Map vs MapPartitions", "occurrences2"],    
        "detected3": ["Serialized Data Formats", "occurrences3"],
        "detected4": ["Avoiding UDFs", "occurrences4"],
    }

    # Iterate over all files in the directory
    for json_file in os.listdir(json_dir):
        if "_All" in json_file and json_file.endswith(".json"):
            json_path = os.path.join(json_dir, json_file)
            with open(json_path, "r") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {json_file}")
                    continue

                # Extract model, file, and process each case
                model = json_file.split("_")[0]
                file_name = json_file.split("_")[-1].replace(".json", ".py")

                for case_key, [case_name, occurrence] in cases_map.items():
                    detected = data.get(case_key, False)
                    occurrences = data.get(occurrence, 0)

                    
                    # Add row for each case
                    csv_data.append([
                        model, case_name, file_name, detected, occurrences
                    ])

    # Write to CSV
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Model", "Case", "File", "Detection", "Occurrences"])
        csv_writer.writerows(csv_data)

    print(f"CSV file created: {output_csv}")

# Directory containing JSON files and output CSV path
json_directory = "output05.01/publishers/meta/models"  # Replace with your JSON files directory
output_csv_path = "detection_results_all_cases05.01.csv"

# Generate CSV
create_csv_from_jsons(json_directory, output_csv_path)
