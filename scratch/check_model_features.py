import xgboost as xgb
import json

MODEL_PATH = "storage/surrogate_xgb.json"

try:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    print(f"Feature names: {model.get_booster().feature_names}")
except Exception as e:
    print(f"Error: {e}")
