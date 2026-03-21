# Megha A - 727823TUAM024

from datetime import datetime
import os
import pandas as pd

print("Roll Number: 727823TUAM024")
print("Timestamp:", datetime.now())

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "..", "data", "Dataset AM024.xlsx")

df = pd.read_excel(file_path, sheet_name="Aggregate")

use_cols = [
    "AEP Adjusted",
    "DFSVX Adjusted",
    "DFLVX Adjusted",
    "FSAGX Adjusted",
    "GS1 (rf)",
    "DFSVX Excess",
    "DFLVX Excess",
    "FSAGX Excess",
    "AEP Excess"
]

data = df[use_cols].dropna()

output_path = os.path.join(base_dir, "..", "data", "prepared_data.csv")
data.to_csv(output_path, index=False)

print("Prepared data saved to:", output_path)
print("Prepared data shape:", data.shape)