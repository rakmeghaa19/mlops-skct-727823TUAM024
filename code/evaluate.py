# Megha A - 727823TUAM024

from datetime import datetime
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Roll Number: 727823TUAM024")
print("Timestamp:", datetime.now())

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "..", "data", "trained_model.pkl")
test_path = os.path.join(base_dir, "..", "data", "test_data.csv")

df = pd.read_csv(test_path)
model = joblib.load(model_path)

X_test = df.drop(columns=["AEP Excess"])
y_test = df["AEP Excess"]

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("Evaluation Results")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)