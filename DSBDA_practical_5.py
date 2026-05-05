# ============================================================
#  PRACTICAL 5 — Logistic Regression
#  Dataset : Social_Network_Ads (created manually - same structure)
#  Goal    : Predict if a user Purchased (1) or Not Purchased (0)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, classification_report)

# ── Create Social_Network_Ads Dataset ───────────────────────
# Same structure as original Kaggle file:
# UserID, Gender, Age, EstimatedSalary, Purchased
np.random.seed(42)
n = 400

age    = np.random.randint(18, 60, n)
salary = np.random.randint(15000, 150000, n)

# Realistic rule: older + higher salary → more likely to purchase
score     = (age - 18) / 42 + (salary - 15000) / 135000
noise     = np.random.normal(0, 0.2, n)
purchased = (score + noise > 0.85).astype(int)

df = pd.DataFrame({
    "UserID"          : range(1, n+1),
    "Gender"          : np.random.choice(["Male", "Female"], n),
    "Age"             : age,
    "EstimatedSalary" : salary,
    "Purchased"       : purchased
})

print("=== Dataset Sample ===")
print(df.head(10))
print("\nShape   :", df.shape)
print("Columns :", list(df.columns))

print("\n=== Class Distribution ===")
print(df["Purchased"].value_counts())
print("0 = Not Purchased | 1 = Purchased")

# ── Features & Target ────────────────────────────────────────
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

# ── Feature Scaling ──────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train-Test Split (75% train, 25% test) ───────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=0)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# ── Logistic Regression Model ────────────────────────────────
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── Confusion Matrix ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print("\n=== Confusion Matrix ===")
print(f"                  Predicted NO   Predicted YES")
print(f"  Actual NO    |     TN = {TN:<5}|     FP = {FP}")
print(f"  Actual YES   |     FN = {FN:<5}|     TP = {TP}")

# ── Evaluation Metrics ───────────────────────────────────────
accuracy   = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision  = precision_score(y_test, y_pred)
recall     = recall_score(y_test, y_pred)

print("\n=== Evaluation Metrics ===")
print(f"  True Positive  (TP) : {TP}")
print(f"  True Negative  (TN) : {TN}")
print(f"  False Positive (FP) : {FP}")
print(f"  False Negative (FN) : {FN}")
print(f"  Accuracy            : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Error Rate          : {error_rate:.4f}  ({error_rate*100:.2f}%)")
print(f"  Precision           : {precision:.4f}")
print(f"  Recall (Sensitivity): {recall:.4f}")

print("\n=== Full Classification Report ===")
print(classification_report(y_test, y_pred,
      target_names=["Not Purchased", "Purchased"]))
