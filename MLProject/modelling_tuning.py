import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
import os, pickle, json
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train = pd.read_csv("train_data_scaled.csv")
test = pd.read_csv("test_data_scaled.csv")

X_train, y_train = train.drop(columns="target"), train["target"]
X_test, y_test = test.drop(columns="target"), test["target"]

# Tracking UI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Skilled_MSML")

# Tuning parameters
params = {"n_estimators": [100, 150], "max_depth": [5, 10]}

with mlflow.start_run():
    # Training dan tuning
    grid = GridSearchCV(RandomForestClassifier(class_weight="balanced", random_state=42), params, cv=3)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    # Manual logging
    mlflow.log_param("best_n_estimators", grid.best_params_["n_estimators"])
    mlflow.log_param("best_max_depth", grid.best_params_["max_depth"])
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="weighted"))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, average="macro"))
    mlflow.sklearn.log_model(grid.best_estimator_, "model")

    # Simpan model lokal (opsional)
    os.makedirs("model_artifacts", exist_ok=True)
    with open("model_artifacts/model.pkl", "wb") as f:
        pickle.dump(grid.best_estimator_, f)

    # 🔽 Tambahan: Artefak Visual & Metadata

    # 1. Confusion matrix PNG
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("training_confusion_matrix.png")
    mlflow.log_artifact("training_confusion_matrix.png")

    # 2. Metrik ringkasan JSON
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="macro")
    }
    with open("metric_info.json", "w") as f:
        json.dump(metrics, f, indent=2)
    mlflow.log_artifact("metric_info.json")

    # 3. Estimator visual HTML
    with open("estimator.html", "w") as f:
        f.write(f"<html><body><h2>Best Estimator</h2><pre>{grid.best_estimator_}</pre></body></html>")
    mlflow.log_artifact("estimator.html")
