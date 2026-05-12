# ============================================================
#  PRACTICAL 6 — Naive Bayes Classification
#  Dataset : Iris Dataset (Built-in sklearn dataset)
#  Goal    : Classify iris flower into 3 species
# ============================================================

import pandas as pd
import numpy as np

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)

# ── Load Iris Dataset from sklearn (Offline & Exam Safe) ───
iris_data = datasets.load_iris(as_frame=True)

# Convert dataset into DataFrame
iris = iris_data.frame

# Add species names
iris["species"] = iris_data.target_names[iris_data.target]

print("=== Dataset Sample ===")
print(iris.head())

print("\n=== Shape of Dataset ===")
print(iris.shape)

print("\n=== Species Present ===")
print(iris["species"].unique())

# ── Features and Target ─────────────────────────────────────
X = iris.drop(columns=["target", "species"])

y = iris["species"]

# ── Train-Test Split (80% Train, 20% Test) ─────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining Samples : {X_train.shape[0]}")

print(f"Testing Samples  : {X_test.shape[0]}")

# ── Build Naive Bayes Model ─────────────────────────────────
# GaussianNB is used because features are continuous values

model = GaussianNB()

model.fit(X_train, y_train)

# ── Predictions ─────────────────────────────────────────────
y_pred = model.predict(X_test)

# ── Confusion Matrix ────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

labels = model.classes_

print("\n=== Confusion Matrix ===")

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)

# ── TP, FP, FN, TN for each class ──────────────────────────
print("\n=== TP, FP, FN, TN per Class ===")

for i, label in enumerate(labels):

    TP = cm[i, i]

    FP = cm[:, i].sum() - TP

    FN = cm[i, :].sum() - TP

    TN = cm.sum() - TP - FP - FN

    print(f"\nClass : {label}")

    print(f"TP = {TP}")

    print(f"FP = {FP}")

    print(f"FN = {FN}")

    print(f"TN = {TN}")

# ── Evaluation Metrics ──────────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)

error_rate = 1 - accuracy

precision = precision_score(
    y_test,
    y_pred,
    average="macro"
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro"
)

print("\n=== Overall Evaluation Metrics ===")

print(f"Accuracy   : {accuracy:.4f}")

print(f"Error Rate : {error_rate:.4f}")

print(f"Precision  : {precision:.4f}")

print(f"Recall     : {recall:.4f}")

# ── Classification Report ──────────────────────────────────
print("\n=== Full Classification Report ===")

print(classification_report(y_test, y_pred))
