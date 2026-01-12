# shock_rate.py
import os
import pandas as pd
import xgboost as xgb
from typing import Optional

def predict_shock(h0_csv_path: str, xgb_model_path: Optional[str] = None):

    if xgb_model_path is None:
        xgb_model_path = os.getenv("XGB_MODEL_PATH", "models/xgb_model.json")

    testing = pd.read_csv(h0_csv_path)

    # 你原本的「drop 第11欄」保留
    testing = testing.drop(testing.columns[10], axis=1)

    model = xgb.Booster()
    model.load_model(xgb_model_path)

    dtest = xgb.DMatrix(testing.values)
    preds = model.predict(dtest)
    return preds  # numpy array

if __name__ == "__main__":
    import sys
    h0 = sys.argv[1] if len(sys.argv) >= 2 else "h0.csv"
    preds = predict_shock(h0)
    print(preds)
