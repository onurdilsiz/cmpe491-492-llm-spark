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

# modelplotter()

def latencyplotter():
    file_path = "all_latency.csv"  # Replace with the correct file name
    data = pd.read_csv(file_path)

    # Ensure Latency is a numeric column
    data['Latency'] = pd.to_numeric(data['Latency']/5, errors='coerce')

    # Plot the latency for each model
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Latency', data=data, palette="Blues", ci=None)

    # Add labels and title
    plt.title('Latency vs Model', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Latency (seconds)', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()

    # Show the plot
    plt.show()

# latencyplotter()

def latencyplotter_grouped():
    file_path = "specific_detection_results.csv"  # Replace with the correct file name
    data = pd.read_csv(file_path)

    # Ensure Latency is a numeric column
    data['Latency'] = pd.to_numeric(data['Latency'], errors='coerce')

    # Create the plot
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Model', y='Latency', hue='Case', data=data, palette="Blues", ci=None)

    # Add labels and title
    plt.title('Latency by Model and Case', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Latency (seconds)', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title='Case', fontsize=10, loc='upper left')
    plt.tight_layout()

    # Show the plot
    plt.show()

# Call the function
# latencyplotter_grouped()


def tokens_plotter_with_input_output():
    # Load the CSV file
    file_path = "cumulative_token_data.csv"  # Replace with your actual file name
    data = pd.read_csv(file_path)

    # Ensure tokens columns are numeric
    data['Input Tokens'] = pd.to_numeric(data['Input Tokens'], errors='coerce')
    data['Output Tokens'] = pd.to_numeric(data['Output Tokens'], errors='coerce')

    # Melt the data to create a long-form DataFrame for grouped plotting
    melted_data = pd.melt(data, 
                          id_vars=["Model"], 
                          value_vars=["Input Tokens", "Output Tokens"], 
                          var_name="Token Type", 
                          value_name="Token Count")

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="Token Count", hue="Token Type", data=melted_data, palette="Blues", ci=None)

    # Add labels and title
    plt.title('Tokens vs Model', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Token Count', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title="Token Type", fontsize=10)
    plt.tight_layout()

    # Show the plot
    plt.show()

# Call the function
# tokens_plotter_with_input_output()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def total_tokens_plotter():
    # Load the CSV file
    file_path = "cumulative_token_data.csv"  # Replace with your actual file name
    data = pd.read_csv(file_path)

    # Ensure Total Tokens column is numeric
    data['Total Tokens'] = pd.to_numeric(data['Total Tokens'], errors='coerce')

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="Total Tokens", data=data, palette="Blues", ci=None)

    # Add labels and title
    plt.title('Total Tokens vs Model', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Total Tokens', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()

    # Show the plot
    plt.show()

# Call the function
# total_tokens_plotter()
def output_tokens_plotter():
    # Load the CSV file
    file_path = "cumulative_token_data.csv"  # Replace with your actual file name
    data = pd.read_csv(file_path)

    # Ensure Total Tokens column is numeric
    data['Output Tokens'] = pd.to_numeric(data['Output Tokens'], errors='coerce')

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="Output Tokens", data=data, palette="Blues", ci=None)

    # Add labels and title
    plt.title('Output Tokens vs Model', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Output Tokens', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()

    # Show the plot
    plt.show()

# Call the function
# output_tokens_plotter()
