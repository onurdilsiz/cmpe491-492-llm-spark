import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# csv_file_path = "all_model_case_success_26.12.csv"



# Read the CSV file into a DataFrame
# data = pd.read_csv(csv_file_path)


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





def calculate_average_tokens(input_file, output_file):
    # Load the CSV file
    data = pd.read_csv(input_file)

    # Ensure Input Tokens and Output Tokens columns are numeric
    data['Input Tokens'] = pd.to_numeric(data['Input Tokens'], errors='coerce')
    data['Output Tokens'] = pd.to_numeric(data['Output Tokens'], errors='coerce')

    # Group by 'Model' and calculate the average input and output tokens
    avg_tokens = data.groupby('Model').agg({
        'Input Tokens': 'mean',
        'Output Tokens': 'mean'
    }).reset_index()

    # Rename the columns for clarity
    avg_tokens.rename(columns={
        'Input Tokens': 'Average Input Tokens',
        'Output Tokens': 'Average Output Tokens'
    }, inplace=True)

    # Write the results to a new CSV file
    avg_tokens.to_csv(output_file, index=False)

    print(f"Average tokens written to {output_file}")

# Example usage
input_file = "cumulative_token_data.csv"  # Replace with your input CSV file name
output_file = "average_tokens_per_model.csv"  # Replace with your desired output CSV file name

# calculate_average_tokens(input_file, output_file)






def calculate_case_specific_averages(input_file, output_file):
    # Load the CSV file
    data = pd.read_csv(input_file)

    # Ensure Input Tokens and Output Tokens columns are numeric
    data['Input Tokens'] = pd.to_numeric(data['Input Tokens'], errors='coerce')
    data['Output Tokens'] = pd.to_numeric(data['Output Tokens'], errors='coerce')

    # Filter for the "All" case and calculate average tokens
    all_case_data = data[data['Case'] == 'All']
    avg_all_case_tokens = all_case_data.groupby('Model').agg({
        'Input Tokens': 'mean',
        'Output Tokens': 'mean'
    }).reset_index()
    avg_all_case_tokens.rename(columns={
        'Input Tokens': 'Avg Input Tokens (All Case)',
        'Output Tokens': 'Avg Output Tokens (All Case)'
    }, inplace=True)

    # Filter for the rest of the cases (excluding "All") and calculate average tokens
    other_cases_data = data[data['Case'] != 'All']
    avg_other_cases_tokens = other_cases_data.groupby('Model').agg({
        'Input Tokens': 'mean',
        'Output Tokens': 'mean'
    }).reset_index()
    avg_other_cases_tokens.rename(columns={
        'Input Tokens': 'Avg Input Tokens (Other Cases)',
        'Output Tokens': 'Avg Output Tokens (Other Cases)'
    }, inplace=True)

    # Merge the results for "All" and "Other" cases
    combined_averages = pd.merge(avg_all_case_tokens, avg_other_cases_tokens, on='Model', how='outer')

    # Write the results to a new CSV file
    combined_averages.to_csv(output_file, index=False)

    print(f"Average tokens per model for 'All' and other cases written to {output_file}")

# Example usage
input_file = "cumulative_token_data.csv"  # Replace with your input CSV file name
output_file = "average_tokens_all_vs_others.csv"  # Replace with your desir

# calculate_case_specific_averages(input_file, output_file)


def plot_average_tokens(input_file):
    # Load the computed averages from the CSV file
    combined_averages = pd.read_csv(input_file)

    # Set up the bar plot
    models = combined_averages['Model']
    avg_input_all = combined_averages['Avg Input Tokens (All Case)']
    avg_output_all = combined_averages['Avg Output Tokens (All Case)']
    avg_input_other = combined_averages['Avg Input Tokens (Other Cases)']
    avg_output_other = combined_averages['Avg Output Tokens (Other Cases)']

    # Define the width of the bars
    bar_width = 0.2
    x = np.arange(len(models))  # Label locations

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot bars for "All Case"
    plt.bar(x - bar_width, avg_input_all, width=bar_width, label='Avg Input Tokens (All Case)', color='lightblue')
    plt.bar(x, avg_output_all, width=bar_width, label='Avg Output Tokens (All Case)', color='blue')

    # Plot bars for "Other Cases"
    plt.bar(x + bar_width, avg_input_other, width=bar_width, label='Avg Input Tokens (Other Cases)', color='pink')
    plt.bar(x + 2 * bar_width, avg_output_other, width=bar_width, label='Avg Output Tokens (Other Cases)', color='red')

    # Add labels, title, and legend
    plt.title('Average Token Counts per Model (All Case vs Other Cases)', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Average Token Count', fontsize=12)
    plt.xticks(x + bar_width / 2, models, rotation=45, fontsize=10)
    plt.legend(title="Token Type", fontsize=10)
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage
input_file = "average_tokens_all_vs_others.csv"  # Replace with the generated CSV file
# plot_average_tokens(input_file)


def calculate_costs_from_averages(token_avg_file, pricing_file, output_file):
    # Load the average token data and pricing data
    token_avg_data = pd.read_csv(token_avg_file)
    pricing_data = pd.read_csv(pricing_file)
    
    # Merge the datasets on the 'Model' column
    merged_data = pd.merge(token_avg_data, pricing_data, on='Model', how='inner')
    
    # Calculate costs for "All Case"
    merged_data['All Case Input Cost'] = (merged_data['Avg Input Tokens (All Case)'] / 1_000_000) * merged_data['Cost Per 1M Token (Input)']
    merged_data['All Case Output Cost'] = (merged_data['Avg Output Tokens (All Case)'] / 1_000_000) * merged_data['Cost Per 1M Token (Output)']
    merged_data['All Case Total Cost'] = merged_data['All Case Input Cost'] + merged_data['All Case Output Cost']
    
    # Calculate costs for "Other Cases"
    merged_data['Other Cases Input Cost'] = (merged_data['Avg Input Tokens (Other Cases)'] / 1_000_000) * merged_data['Cost Per 1M Token (Input)']
    merged_data['Other Cases Output Cost'] = (merged_data['Avg Output Tokens (Other Cases)'] / 1_000_000) * merged_data['Cost Per 1M Token (Output)']
    merged_data['Other Cases Total Cost'] = merged_data['Other Cases Input Cost'] + merged_data['Other Cases Output Cost']
    
    # Save the results to a new CSV file
    merged_data.to_csv(output_file, index=False)
    print(f"Costs calculated and saved to {output_file}")

# Example usage
token_avg_file = "average_tokens_all_vs_others.csv"  # Replace with your token averages file
pricing_file = "model_pricings.csv"  # Replace with your pricing data file
output_file = "calculated_costs.csv"  # Replace with the desired output file name

calculate_costs_from_averages(token_avg_file, pricing_file, output_file)

def plot_costs(cost_file):
    # Load the calculated cost data
    cost_data = pd.read_csv(cost_file)

    
    # Melt the data for grouped plotting
    melted_data = pd.melt(
        cost_data, 
        id_vars=["Model"], 
        value_vars=["All Case Total Cost", "Other Cases Total Cost"], 
        var_name="Case Type", 
        value_name="Average Cost"
    )
    
    # Plot total costs
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="Average Cost", hue="Case Type", data=melted_data, palette="Blues")
    plt.title('Average Costs Per Model for All Case and Separate Cases', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Average Cost ($)', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title="Case Type")
    plt.tight_layout()
    plt.show()

# Example usage
# plot_costs(output_file)




def plot_accuracy_cost_ratio(accuracy_file, cost_file):
    # Load accuracy and cost data
    accuracy_data = pd.read_csv(accuracy_file)
    cost_data = pd.read_csv(cost_file)

    # Ensure column names match, and select the relevant cost column (Other Cases Total Cost)
    cost_data.rename(columns={"Other Cases Total Cost": "Total Cost (Other)"}, inplace=True)
    
    # Merge the accuracy and cost data on the "Model" column
    merged_data = pd.merge(accuracy_data, cost_data[["Model", "Total Cost (Other)"]], on="Model")
    
    # Calculate Accuracy/Cost ratio
    merged_data["Accuracy/Cost"] = merged_data["Accuracy"] / merged_data["Total Cost (Other)"]
    
    # Plot Accuracy/Cost vs Model
    plt.figure(figsize=(10, 6))
    sns.barplot(data=merged_data, x="Model", y="Accuracy/Cost", palette="Blues")
    plt.title("Accuracy/Cost Ratio vs Model", fontsize=16)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Accuracy/Cost Ratio", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.show()

# Example usage
accuracy_file = "specific_metrics_per_model.csv"  # Replace with your accuracy file
cost_file = "calculated_costs.csv"  # Replace with your cost file

# plot_accuracy_cost_ratio(accuracy_file, cost_file)





def plot_accuracy(file_path):
    # Load the specific metrics CSV file
    metrics_data = pd.read_csv(file_path)

    # Ensure the Accuracy column is numeric
    metrics_data['Accuracy'] = pd.to_numeric(metrics_data['Accuracy'], errors='coerce')

    # Plot Accuracy vs Model
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_data, x="Model", y="Accuracy", palette="Blues")
    plt.title("Accuracy vs Model", fontsize=16)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage
file_path = "analysisfiles/specific_metrics_per_model.csv"  # Replace with your file name
# plot_accuracy(file_path)


def plot_accuracy_grouped_by_case(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Filter the data to include only the Accuracy column
    accuracy_data = data[["Case", "Model", "Accuracy"]]

    # Sort the data alphabetically by Case and Model
    accuracy_data = accuracy_data.sort_values(by=["Case", "Model"], ascending=[True, True])

    # Create the grouped bar plot
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=accuracy_data,
        x="Case",
        y="Accuracy",
        hue="Model",
        ci=None,
        palette="Blues"
    )

    # Customize the plot
    plt.title("Accuracy Grouped by Case and Model", fontsize=16)
    plt.xlabel("Case", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title="Model", fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage
file_path = "specific_metrics_per_case_model.csv"  # Replace with your actual file name
# plot_accuracy_grouped_by_case(file_path)




def plot_accuracy_with_scenarios(file1, file2, scenario1_name, scenario2_name):
    """
    Plot accuracy with two scenarios from two files.

    :param file1: Path to the first file for Scenario 1
    :param file2: Path to the second file for Scenario 2
    :param scenario1_name: Name for the first scenario
    :param scenario2_name: Name for the second scenario
    """
    # Load the CSV files
    metrics_data_1 = pd.read_csv(file1)
    metrics_data_2 = pd.read_csv(file2)

    # Ensure the Accuracy column is numeric
    metrics_data_1['Accuracy'] = pd.to_numeric(metrics_data_1['Accuracy'], errors='coerce')
    metrics_data_2['Accuracy'] = pd.to_numeric(metrics_data_2['Accuracy'], errors='coerce')

    # Add a column to indicate the scenario
    metrics_data_1['Scenario'] = scenario1_name
    metrics_data_2['Scenario'] = scenario2_name

    # Combine the two dataframes
    combined_data = pd.concat([metrics_data_1, metrics_data_2])

    # Plot Accuracy vs Model grouped by Scenario
    plt.figure(figsize=(14, 8))
    sns.barplot(data=combined_data, x="Model", y="Accuracy", hue="Scenario", palette="Blues", ci=None)
    
    # Customize the plot
    plt.title("Accuracy vs Model Across Scenarios", fontsize=16)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title="Scenario", fontsize=10, loc="upper left")
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage
file1 = "analysisfiles/whole_metrics_per_model.csv"  # Replace with the path to the first file
file2 = "analysisfiles/specific_metrics_per_model.csv"  # Replace with the path to the second file
scenario1_name = "All"  # Replace with the name of the first scenario
scenario2_name = "Separate"  # Replace with the name of the second scenario

plot_accuracy_with_scenarios(file1, file2, scenario1_name, scenario2_name)