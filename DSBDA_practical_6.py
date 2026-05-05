# ============================================================
#  PRACTICAL 6 — Naive Bayes Classification
#  Dataset : Iris
#  Source  : https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
#  Goal    : Classify iris flower into 3 species
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, classification_report)

# ── Load Dataset ─────────────────────────────────────────────
URL  = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df   = pd.read_csv(URL)

print("=== Dataset Sample ===")
print(df.head())
print("\nShape  :", df.shape)
print("Species:", df["species"].unique())

# ── Features and Target ──────────────────────────────────────
X = df.drop(columns=["species"])
y = df["species"]

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Naive Bayes Model ────────────────────────────────────────
# GaussianNB is used because features are continuous (float values)
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ── Confusion Matrix ─────────────────────────────────────────
cm     = confusion_matrix(y_test, y_pred)
labels = model.classes_

print("\n=== Confusion Matrix ===")
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("         Predicted")
print(cm_df)

# ── TP, FP, TN, FN per class (One-vs-Rest) ──────────────────
print("\n=== TP, FP, TN, FN per Class (One-vs-Rest) ===")
for i, label in enumerate(labels):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - TP - FP - FN
    print(f"\n  Class: {label}")
    print(f"    TP={TP}  FP={FP}  FN={FN}  TN={TN}")

# ── Overall Metrics ──────────────────────────────────────────
accuracy   = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision  = precision_score(y_test, y_pred, average="macro")
recall     = recall_score(y_test, y_pred, average="macro")

print("\n=== Overall Evaluation Metrics ===")
print(f"  Accuracy            : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Error Rate          : {error_rate:.4f}  ({error_rate*100:.2f}%)")
print(f"  Precision (macro)   : {precision:.4f}")
print(f"  Recall    (macro)   : {recall:.4f}")

print("\n=== Full Classification Report ===")
print(classification_report(y_test, y_pred))
