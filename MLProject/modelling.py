import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Aktifkan autolog agar semua param, metrik, model tercatat otomatis
mlflow.sklearn.autolog()

# Baca data hasil preprocessing dari folder lokal
train = pd.read_csv("train_data_scaled.csv")
test = pd.read_csv("test_data_scaled.csv")

# Pisah fitur dan target
X_train, y_train = train.drop(columns="target"), train["target"]
X_test, y_test = test.drop(columns="target"), test["target"]

# Mulai run experiment lokal (Tracking URI default: localhost)
with mlflow.start_run(run_name="basic_autolog_rf"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    # Tidak perlu log_model lagi, karena autolog sudah mencatat otomatis