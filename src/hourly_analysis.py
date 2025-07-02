import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

def plot_hourly_fraud():
    df = pd.read_csv('data/paysim.csv')
    df = df.sample(n=100000, random_state=42)
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]
    df['hour'] = df['step'] % 24
    hourly_fraud_rate = df.groupby('hour')['isFraud'].mean()
    plt.figure(figsize=(8, 5))
    plt.plot(hourly_fraud_rate.index, hourly_fraud_rate.values, marker='o')
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Fraud Rate")
    plt.title("Fraud Trend by Hour")
    plt.grid(True)
    plt.savefig("outputs/hourly_fraud_trend.png")

if __name__ == "__main__":
    plot_hourly_fraud()
