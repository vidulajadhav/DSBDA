# ============================================================
# PRACTICAL 5 — Logistic Regression
# Dataset : Social_Network_Ads (Generated & Saved Locally)
# Goal    : Predict Purchased (1) or Not Purchased (0)
# ============================================================

# ── Import Libraries ────────────────────────────────────────
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)

# ============================================================
# CREATE DATASET USING NUMPY RANDOM
# ============================================================

np.random.seed(42)

n = 100

age = np.random.randint(18, 60, n)

salary = np.random.randint(15000, 150000, n)

# Realistic Purchase Logic
score = (age - 18) / 42 + (salary - 15000) / 135000

noise = np.random.normal(0, 0.2, n)

purchased = (score + noise > 0.85).astype(int)

# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({

    "UserID": range(1, n + 1),

    "Gender": np.random.choice(
        ["Male", "Female"], n
    ),

    "Age": age,

    "EstimatedSalary": salary,

    "Purchased": purchased
})

# ============================================================
# ADD FEW MISSING VALUES
# ============================================================

df.loc[5, "Age"] = np.nan

df.loc[12, "EstimatedSalary"] = np.nan

df.loc[25, "Age"] = np.nan

# ============================================================
# SAVE DATASET LOCALLY AS CSV FILE
# ============================================================

df.to_csv("Social_Network_Ads.csv", index=False)

print("Dataset created and saved successfully!")

# ============================================================
# LOAD DATASET FROM CSV FILE
# ============================================================

df = pd.read_csv("Social_Network_Ads.csv")

# ============================================================
# DISPLAY DATASET
# ============================================================

print("\n=== Dataset Sample ===")

print(df.head(10))

print("\nShape :", df.shape)

print("\nColumns :", list(df.columns))

# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\n=== Missing Values Before Handling ===")

print(df.isnull().sum())

# ============================================================
# HANDLE MISSING VALUES
# ============================================================

df.fillna(df.mean(numeric_only=True), inplace=True)

print("\n=== Missing Values After Handling ===")

print(df.isnull().sum())

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n=== Class Distribution ===")

print(df["Purchased"].value_counts())

print("\n0 = Not Purchased")
print("1 = Purchased")

# ============================================================
# FEATURES & TARGET
# ============================================================

X = df[["Age", "EstimatedSalary"]]

y = df["Purchased"]

# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.25,
    random_state=0
)

print(f"\nTraining Samples : {X_train.shape[0]}")

print(f"Testing Samples  : {X_test.shape[0]}")

# ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(random_state=0)

model.fit(X_train, y_train)

# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("\n=== Confusion Matrix ===")

print(f"                  Predicted NO   Predicted YES")

print(f"  Actual NO    |     TN = {TN:<5}|     FP = {FP}")

print(f"  Actual YES   |     FN = {FN:<5}|     TP = {TP}")

# ============================================================
# EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

error_rate = 1 - accuracy

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

print("\n=== Evaluation Metrics ===")

print(f"True Positive  (TP) : {TP}")

print(f"True Negative  (TN) : {TN}")

print(f"False Positive (FP) : {FP}")

print(f"False Negative (FN) : {FN}")

print(f"Accuracy            : {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"Error Rate          : {error_rate:.4f} ({error_rate*100:.2f}%)")

print(f"Precision           : {precision:.4f}")

print(f"Recall              : {recall:.4f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n=== Full Classification Report ===")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Purchased", "Purchased"]
))
