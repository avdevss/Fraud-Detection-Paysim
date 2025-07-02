import os
import shap
import xgboost
import pandas as pd
from preprocess import load_data
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

def interpret_model():
    X_train, X_test, y_train, y_test = load_data()
    model = xgboost.XGBClassifier()
    model.load_model("models/xgb_paysim.json")
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png")

if __name__ == "__main__":
    interpret_model()
