import os
import joblib
import pandas as pd
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Load data
df = pd.read_csv("stroke_preprocessed.csv")

# Fitur dan target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Autolog MLflow
mlflow.sklearn.autolog()

# Model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

# Training
model.fit(X_train, y_train)

# Simpan model
os.makedirs("model", exist_ok=True)

model_path = "model/model.pkl"

joblib.dump(
    model,
    model_path
)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Accuracy: {accuracy:.4f}")

# Classification Report
report = classification_report(
    y_test,
    y_pred
)

with open(
    "classification_report.txt",
    "w"
) as f:
    f.write(report)

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)

plt.savefig(
    "confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()

print("\nClassification Report:")
print(report)

print("\nTraining selesai.")