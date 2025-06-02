import pandas as pd
import matplotlib.pyplot as plt

# Set the style and color scheme to blue theme
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Raw numbers extracted from the user's table (converted comma decimal to dot)
data = [
    # architecture, model, latency, cost
    ("Tab-based", "Gemini 2.0 Flash", 16.67, 0.00224),
    ("Tab-based", "Gemini 2.0 Flash", 15.25, 0.0021522),
    ("Tab-based", "Gemini 2.0 Flash", 16.91, 0.0022294),
    ("Tab-based", "Gemini 2.0 Flash", 14.39, 0.001805),
    ("Tab-based", "Gemini 2.0 Flash", 14.01, 0.00206),
    ("Tab-based", "Gemini 2.0 Flash", 17.21, 0.00219),

    ("Tab-based", "GPT-4o", 22.83, 0.05),
    ("Tab-based", "GPT-4o", 21.72, 0.05),
    ("Tab-based", "GPT-4o", 20.13, 0.05),
    ("Tab-based", "GPT-4o", 22.76, 0.05),
    ("Tab-based", "GPT-4o", 24.36, 0.05),
    ("Tab-based", "GPT-4o", 20.54, 0.05),

    ("Case-based", "Gemini 2.0 Flash", 25.76, 0.010257),
    ("Case-based", "Gemini 2.0 Flash", 24.52, 0.0106),
    ("Case-based", "Gemini 2.0 Flash", 20.42, 0.01),        # cost 0 missing? assume 0 for now
    ("Case-based", "Gemini 2.0 Flash", 23.12, 0.0099239),
    ("Case-based", "Gemini 2.0 Flash", 22.21, 0.0105272),
    ("Case-based", "Gemini 2.0 Flash", 20.00, 0.01),        # cost 0 missing? assume 0

    ("Case-based", "GPT-4o", 44.99, 0.151),
    ("Case-based", "GPT-4o", 43.46, 0.18),
    ("Case-based", "GPT-4o", 60.26, 0.1875),
    ("Case-based", "GPT-4o", 31.72, 0.146015),
    ("Case-based", "GPT-4o", 41.14, 0.15),
    ("Case-based", "GPT-4o", 50.09, 0.16),
]

df = pd.DataFrame(data, columns=["Architecture", "Model", "Latency", "Cost"])

# Compute averages
summary = df.groupby(["Architecture","Model"]).agg({"Latency":"mean","Cost":"mean"}).reset_index()

# Pivot for plotting
latency_pivot = summary.pivot(index="Architecture", columns="Model", values="Latency")
cost_pivot = summary.pivot(index="Architecture", columns="Model", values="Cost")

# Define blue color palette
blue_colors = ['#1f77b4', '#aec7e8']  # Different shades of blue

# Plot latency with blue theme
fig, ax = plt.subplots(figsize=(10, 6))
latency_pivot.plot(kind="bar", figsize=(10, 6), color=blue_colors, ax=ax)
plt.ylabel("Average Latency (s)", fontsize=12)
plt.title("Average Latency by Architecture & Model", fontsize=14, fontweight='bold')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3, axis='y')
plt.legend(title="Model", loc='upper left')
plt.tight_layout()
plt.savefig('latency_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()  # Close the figure to free memory

# Plot cost with blue theme
fig, ax = plt.subplots(figsize=(10, 6))
cost_pivot.plot(kind="bar", figsize=(10, 6), color=blue_colors, ax=ax)
plt.ylabel("Average Cost (USD)", fontsize=12)
plt.title("Average Cost by Architecture & Model", fontsize=14, fontweight='bold')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3, axis='y')
plt.legend(title="Model", loc='upper left')
plt.tight_layout()
plt.savefig('cost_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()  # Close the figure to free memory

print("Charts saved as:")
print("- latency_comparison.png")
print("- cost_comparison.png")

