import pandas as pd
import argparse
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

def log_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    os.makedirs("artifacts", exist_ok=True)
    fig_path = f"artifacts/cm_{model_name}.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)
    plt.close()

def train_and_log_model(X_train, X_test, y_train, y_test, params, model_count):
    model = RandomForestClassifier(**params)
    run_name = f"RF_{model_count}"
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        log_confusion_matrix(y_test, y_pred, run_name)
        mlflow.sklearn.log_model(model, "model")

        return f1, model

def main(data_path):
    df = pd.read_csv(data_path)
    X = df.drop("Air Quality", axis=1)
    y = df["Air Quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Parameter grid untuk Grid Search manual
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
        "random_state": [42]
    }

    param_combos = list(product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["random_state"]
    ))

    best_f1 = 0
    best_model = None
    best_params = None

    for i, (n_est, depth, seed) in enumerate(param_combos):
        params = {
            "n_estimators": n_est,
            "max_depth": depth,
            "random_state": seed
        }
        f1, model = train_and_log_model(X_train, X_test, y_train, y_test, params, i)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = params

    # Registrasi model terbaik
    if best_model:
        with mlflow.start_run(run_name="Best_RF_Model"):
            mlflow.log_params(best_params)
            mlflow.log_metric("best_f1_score", best_f1)
            y_pred_best = best_model.predict(X_test)
            log_confusion_matrix(y_test, y_pred_best, "Best_RF_Model")
            mlflow.sklearn.log_model(
                best_model, "model", registered_model_name="BestRFModel"
        )
            log_confusion_matrix(y_test, y_pred_best, "Best_RF_Model")
            print(f"Model terbaik diregistrasi 🎯 | F1 Score: {best_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
