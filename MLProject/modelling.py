import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Argument untuk folder data
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="dataset_preprocessing")
args = parser.parse_args()

# Baca data
train = pd.read_csv(f"{args.data_path}/train_data_scaled.csv")
test = pd.read_csv(f"{args.data_path}/test_data_scaled.csv")

X_train, y_train = train.drop(columns="target"), train["target"]
X_test, y_test = test.drop(columns="target"), test["target"]

# Autolog MLflow
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="autolog_random_forest"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
