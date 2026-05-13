#DATA ANALYTICS PRACTICAL - Logistic Regression
#Dataset : Social Network Ads (Self-Created → Social_Network_Ads.csv)
#Goal    : Predict Purchased (1) or Not Purchased (0)

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

#STEP 0 : CREATE DATASET AND SAVE TO CSV
def create_social_network_dataset(filename="Social_Network_Ads.csv",
                                   n=100, seed=42):
    """
    Creates a realistic Social Network Ads dataset with:
    - 100 entries using numpy random
    - Features : UserID, Gender, Age, EstimatedSalary
    - Target   : Purchased (0 = No, 1 = Yes)
    - Realistic logic: older + higher salary → more likely to purchase
    - Injected missing values in Age and EstimatedSalary
    Saves to CSV file.
    """
    np.random.seed(seed)

    #Generate Features 
    age    = np.random.randint(18, 60, n)
    salary = np.random.randint(15000, 150000, n)
    gender = np.random.choice(["Male", "Female"], n, p=[0.50, 0.50])

    #Realistic Purchase Logic 
    #Normalize age and salary to 0-1 range
    #Higher age + higher salary → higher purchase score
    age_norm    = (age - 18) / (60 - 18)
    salary_norm = (salary - 15000) / (150000 - 15000)

    #Combined score with noise
    score     = 0.5 * age_norm + 0.5 * salary_norm
    noise     = np.random.normal(0, 0.15, n)
    purchased = (score + noise > 0.55).astype(int)

    df = pd.DataFrame({
        "UserID"         : range(1, n + 1),
        "Gender"         : gender,
        "Age"            : age.astype(float),
        "EstimatedSalary": salary.astype(float),
        "Purchased"      : purchased,
    })

    #Inject Missing Values: Only in feature columns — NEVER in target column
    missing_age    = np.random.choice(n, 5, replace=False)
    missing_salary = np.random.choice(n, 4, replace=False)

    df.loc[missing_age,    "Age"]             = np.nan
    df.loc[missing_salary, "EstimatedSalary"] = np.nan

    #Save to CSV 
    df.to_csv(filename, index=False)
    print(f"Dataset created and saved to '{filename}'")
    print(f"   Rows: {len(df)} | Columns: {list(df.columns)}")
    print(f"   Purchase distribution — "
          f"Bought: {purchased.sum()} | "
          f"Not Bought: {n - purchased.sum()}")

#Generate and Save 
create_social_network_dataset("Social_Network_Ads.csv", n=100)

#STEP 1 : LOAD DATASET
df = pd.read_csv("Social_Network_Ads.csv")

print("\n=== Dataset Sample ===")
print(df.head(10))

print("\nShape   :", df.shape)
print("Columns :", list(df.columns))

#Feature Descriptions
print("\n=== Feature Descriptions ===")
feat_desc = {
    "UserID"         : "Unique user identifier",
    "Gender"         : "Male / Female (categorical)",
    "Age"            : "Age of the user in years",
    "EstimatedSalary": "Estimated annual salary in USD",
    "Purchased"      : "0 = Not Purchased | 1 = Purchased (TARGET)",
}
for col, desc in feat_desc.items():
    print(f"  {col:16s} : {desc}")

#STEP 2 : CHECK AND HANDLE MISSING VALUES
print("\n=== Missing Values Before Handling ===")
print(df.isnull().sum())

#Fill ONLY feature columns with mean
#NEVER fill target column (Purchased)
df["Age"]             = df["Age"].fillna(df["Age"].mean()).round(1)
df["EstimatedSalary"] = df["EstimatedSalary"].fillna(df["EstimatedSalary"].mean()).round(2)

print("\n=== Missing Values After Handling ===")
print(df.isnull().sum())

#STEP 3 : CLASS DISTRIBUTION
print("\n=== Class Distribution ===")
print(df["Purchased"].value_counts())
print("\n  0 = Not Purchased")
print("  1 = Purchased")

#STEP 4 : FEATURES AND TARGET, Only Age and EstimatedSalary used as features
#UserID  → just a serial number, no predictive value
#Gender  → categorical, would need encoding (kept simple here)
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

print(f"\nFeatures (X) shape : {X.shape}")
print(f"Target   (y) shape : {y.shape}")

#STEP 5 : FEATURE SCALING (Standardization), Age: 18-60, Salary: 15000-150000, Without scaling, salary dominates — scaling fixes this
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

#STEP 6 : TRAIN-TEST SPLIT (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.25,
    random_state=0
)

print(f"\nTraining Samples : {X_train.shape[0]}")
print(f"Testing  Samples : {X_test.shape[0]}")

#STEP 7 : BUILD LOGISTIC REGRESSION MODEL
#Logistic Regression uses Sigmoid function: P(y=1) = 1 / (1 + e^(-z)); If P > 0.5 → Purchased = 1, else Purchased = 0
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

print("\n=== Model Coefficients ===")
coeff_df = pd.DataFrame({
    "Feature"    : ["Age", "EstimatedSalary"],
    "Coefficient": model.coef_[0].round(4)
})
print(coeff_df.to_string(index=False))
print(f"Intercept : {model.intercept_[0]:.4f}")

#STEP 8 : PREDICTIONS
y_pred = model.predict(X_test)

print("\n=== Actual vs Predicted (first 10) ===")
comparison = pd.DataFrame({
    "Actual"   : y_test.values[:10],
    "Predicted": y_pred[:10]
})
print(comparison.to_string(index=False))

#STEP 9 : CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print("\n=== Confusion Matrix ===")
print(f"                  Predicted NO    Predicted YES")
print(f"  Actual NO    |    TN = {TN:<6} |    FP = {FP}")
print(f"  Actual YES   |    FN = {FN:<6} |    TP = {TP}")

#What each value means 
print("\n=== Confusion Matrix Terms ===")
print(f"  TP (True Positive)  : {TP} → Predicted Purchased,  Actually Purchased ")
print(f"  TN (True Negative)  : {TN} → Predicted Not Bought, Actually Not Bought ")
print(f"  FP (False Positive) : {FP} → Predicted Purchased,  Actually Not Bought ")
print(f"  FN (False Negative) : {FN} → Predicted Not Bought, Actually Purchased  ")

#STEP 10 : EVALUATION METRICS
accuracy   = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision  = precision_score(y_test, y_pred)
recall     = recall_score(y_test, y_pred)

print("\n=== Evaluation Metrics ===")
print(f"  Accuracy   = (TP+TN)/(TP+TN+FP+FN) = {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Error Rate = 1 - Accuracy           = {error_rate:.4f} ({error_rate*100:.2f}%)")
print(f"  Precision  = TP/(TP+FP)             = {precision:.4f}")
print(f"  Recall     = TP/(TP+FN)             = {recall:.4f}")

#STEP 11 : FULL CLASSIFICATION REPORT
print("\n=== Full Classification Report ===")
print(classification_report(
    y_test, y_pred,
    target_names=["Not Purchased", "Purchased"]
))
