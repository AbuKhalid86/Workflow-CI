import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import os
from datetime import datetime

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default=".")
parser.add_argument("--save_path", type=str, default="saved_artifacts")
args = parser.parse_args()

# Buat folder artefak jika belum ada
os.makedirs(args.save_path, exist_ok=True)

# Baca data
train = pd.read_csv(f"{args.data_path}/train_data_scaled.csv")
test = pd.read_csv(f"{args.data_path}/test_data_scaled.csv")
X_train, y_train = train.drop(columns="target"), train["target"]
X_test, y_test = test.drop(columns="target"), test["target"]

# Autolog MLflow
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="autolog_rf_ci"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Simpan log timestamp sebagai artefak
    with open(f"{args.save_path}/log.txt", "w") as f:
        f.write(f"Training sukses pada {datetime.now()}\n")
