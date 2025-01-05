import pandas as pd

# Load the dataset
results_df = pd.read_csv('detection_results_all_cases26.12.csv')

# Add a new column that checks for both conditions (Diff of Occurrences = 0 and Success = TRUE)
results_df['Combined_Success'] = (results_df['Diff of Occurrences'] == 0) & (results_df['Success'] == True)

# Calculate success rate for each model based on the new condition
model_success_rate = results_df.groupby('Model')['Combined_Success'].mean() * 100

# Calculate success rate for each model-case combination based on the new condition
model_case_success_rate = results_df.groupby(['Model', 'Case'])['Combined_Success'].mean() * 100

# Calculate success rate for each specific case based on the new condition
case_success_rate = results_df.groupby('Case')['Combined_Success'].mean() * 100

# Save the results to CSV files
model_success_rate.to_csv("all_model_combined_success_26.12.csv", header=["Success Rate"])
case_success_rate.to_csv("all_case_combined_success_26.12.csv", header=["Success Rate"])
model_case_success_rate.to_csv("all_model_case_combined_success_26.12.csv", header=["Success Rate"])

print("\nCombined success rates have been calculated and saved to CSV files.")
