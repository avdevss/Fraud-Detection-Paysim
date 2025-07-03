import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

# Load preprocessed data
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "paysim_sample.csv")
data_path = os.path.abspath(data_path)
df = pd.read_csv(data_path)

# Features and target
features = ['amount', 'errorOrig', 'errorDest', 'hour_sin', 'hour_cos']
X = df[features]
y = df['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Define XGBoost model
model = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# 🔧 Threshold tuning section
threshold = 0.8  # Try changing this to 0.6, 0.75 etc.
y_pred_thresh = (y_proba >= threshold).astype(int)

print(f"\n🧪 Evaluation at Threshold = {threshold}")
print(classification_report(y_test, y_pred_thresh, target_names=["Not Fraud", "Fraud"]))

# ROC AUC (threshold-independent)
roc_auc = roc_auc_score(y_test, y_proba)
print(f"✅ ROC AUC Score: {roc_auc:.3f}")

# Save the trained model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "xgb_paysim.json"))
model.save_model(model_path)
