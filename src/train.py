import os
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from preprocess import load_data

os.makedirs("models", exist_ok=True)

def train_model():
    X_train, X_test, y_train, y_test = load_data()
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
    model.save_model("models/xgb_paysim.json")

if __name__ == "__main__":
    train_model()
