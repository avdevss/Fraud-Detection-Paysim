import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filepath='data/paysim.csv'):
    df = pd.read_csv(filepath)
    df = df.sample(n=100000, random_state=42)
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
    df['errorOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['hour'] = df['step'] % 24
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    features = ['amount', 'errorOrig', 'errorDest', 'hour_sin', 'hour_cos']
    X = df[features]
    y = df['isFraud']
    return train_test_split(X, y, test_size=0.2, random_state=42)
