import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


csv_file_path = "all_model_case_success_26.12.csv"



# Read the CSV file into a DataFrame
data = pd.read_csv(csv_file_path)


def case_modelplotter(data):
    plt.figure(figsize=(15, 8))
    sns.barplot(data=data, x="Case", y="Success Rate", hue="Model", dodge=True, palette="Blues")
    plt.title("Success Rate by Model and Case", fontsize=16)
    plt.xlabel("Case", fontsize=12)
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title="Model", fontsize=10, loc="upper left")
    plt.tight_layout()
    plt.show()


def model_caseplotter(data):
    plt.figure(figsize=(15, 8))
    sns.barplot(data=data, x="Model", y="Success Rate", hue="Case", dodge=True, palette="Blues")
    plt.title("Success Rate by Model and Case", fontsize=16)
    plt.xlabel("Case", fontsize=12)
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title="Model", fontsize=10, loc="upper left")
    plt.tight_layout()
    plt.show()

import numpy as np


def modelplotter():
    # Load the two CSV files
    file1 = 'all_model_success_26.12.csv'  # Replace with the first file's name
    file2 = 'specific_model_success_26.12.csv'  # Replace with the second file's name

    # Read the data
    scenario1 = pd.read_csv(file1)
    scenario2 = pd.read_csv(file2)

    # Add a column to indicate the scenario
    scenario1['Scenario'] = 'All'
    scenario2['Scenario'] = 'Separate'
    combined_data = pd.concat([scenario1, scenario2])


    # Get unique models and scenarios
    models = combined_data['Model'].unique()
    scenarios = combined_data['Scenario'].unique()

    # Set bar width and positions
    x = np.arange(len(models))  # Position of models on x-axis
    width = 0.35  # Width of bars

    # Create a plot
    plt.figure(figsize=(12, 6))
    
    # Plot each scenario's bars
    for i, scenario in enumerate(scenarios):
        scenario_data = combined_data[combined_data['Scenario'] == scenario]
        # Match the model order
        scenario_data = scenario_data.set_index('Model').reindex(models).reset_index()
        plt.bar(x + i * width, scenario_data['Success Rate'], width, label=scenario
                )

    # Add labels and title
    plt.title('Comparison of Success Rates Across Models and Scenarios', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.xticks(x + width / 2, models, rotation=45, fontsize=10)  # Center the ticks
    plt.legend(title='Scenario')
    plt.tight_layout()
    plt.show()

modelplotter()