print("✅ Script started")

import pandas as pd
import os

# Check if file exists
file_path = "data/paysim.csv"
if not os.path.exists(file_path):
    print(f"❌ File not found at: {file_path}")
    exit()

print("📂 File found. Reading CSV...")

df = pd.read_csv(file_path)
df_sample = df.sample(n=100000, random_state=42)
df_sample.to_csv("data/paysim_sample.csv", index=False)

print("✅ Sample saved to data/paysim_sample.csv")

