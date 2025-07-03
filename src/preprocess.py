import pandas as pd
import numpy as np
import os

# Construct absolute path to paysim.csv
script_dir = os.path.dirname(__file__)
data_path = os.path.abspath(os.path.join(script_dir, "..", "data", "paysim.csv"))

# Try loading just the first 200,000 rows
try:
    df = pd.read_csv(data_path, nrows=200000)
except FileNotFoundError:
    exit()
except pd.errors.ParserError as e:
    exit()

# Filter only 'TRANSFER' and 'CASH_OUT' transaction types
df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()

# Feature engineering
df['errorOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
df['errorDest'] = df['newbalanceDest'] - df['oldbalanceDest'] - df['amount']
df['hour'] = df['step'] % 24
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Save the processed sample
output_path = os.path.abspath(os.path.join(script_dir, "..", "data", "paysim_sample.csv"))
df.to_csv(output_path, index=False)

