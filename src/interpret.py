import pandas as pd
import shap
import os
import xgboost as xgb
import matplotlib.pyplot as plt

# Load data
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "paysim_sample.csv"))
df = pd.read_csv(data_path)

# Features and model
features = ['amount', 'errorOrig', 'errorDest', 'hour_sin', 'hour_cos']
X = df[features]

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "xgb_paysim.json"))
model = xgb.XGBClassifier()
model.load_model(model_path)

# SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(X)

# SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X, show=False)
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "shap_summary.png"))
plt.savefig(output_path, bbox_inches='tight')
