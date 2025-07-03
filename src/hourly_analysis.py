import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load sampled data
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "paysim_sample.csv"))
df = pd.read_csv(data_path)

# Extract hour from 'step' column (each step = 1 hour)
df['hour'] = df['step'] % 24

# Fraud rate by hour
hourly = df.groupby('hour')['isFraud'].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(x='hour', y='isFraud', data=hourly, marker='o')
plt.title("Average Fraud Rate by Hour of Day")
plt.xlabel("Hour (0-23)")
plt.ylabel("Fraud Rate")
plt.grid(True)

# Save
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "hourly_fraud_trend.png"))
plt.savefig(output_path)
