# Megha A - 727823TUAM024

from datetime import datetime
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

print("Roll Number: 727823TUAM024")
print("Timestamp:", datetime.now())

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "..", "data", "prepared_data.csv")

df = pd.read_csv(file_path)

X = df.drop(columns=["AEP Excess"])
y = df["AEP Excess"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

model_path = os.path.join(base_dir, "..", "data", "trained_model.pkl")
test_path = os.path.join(base_dir, "..", "data", "test_data.csv")

joblib.dump(model, model_path)

test_df = X_test.copy()
test_df["AEP Excess"] = y_test
test_df.to_csv(test_path, index=False)

print("Model saved to:", model_path)
print("Test data saved to:", test_path)