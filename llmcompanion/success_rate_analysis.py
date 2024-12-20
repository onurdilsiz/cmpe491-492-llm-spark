import pandas as pd
# Reimport the dataset to ensure no duplicate columns
results_df = pd.read_csv('detectionresults3.csv')

# Calculate success rate for each model
model_success_rate = results_df.groupby('Model')['Success'].mean() * 100

# Calculate success rate for each model-case combination
model_case_success_rate = results_df.groupby(['Model', 'Case'])['Success'].mean() * 100

# Calculate success rate for each specific case
case_success_rate = results_df.groupby('Case')['Success'].mean() * 100

# Save results to CSV
model_success_rate.to_csv("specific_model_success.csv", header=["Success Rate"])
case_success_rate.to_csv("specific_case_success.csv", header=["Success Rate"])
model_case_success_rate.to_csv("specific_model_case_success.csv", header=["Success Rate"])

print("\nSuccess rates have been calculated and saved to CSV files.")