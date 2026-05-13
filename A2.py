#DATA SCIENCE PRACTICAL - Data Wrangling II
#Dataset : Academic Performance (Self-Created → academic_performance.csv

#Q1. Import Libraries 
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

#STEP 0 : CREATE DATASET AND SAVE TO CSV
def create_academic_dataset(filename="academic_performance.csv", n=60, seed=42):
    """
    Creates a realistic academic performance dataset with:
    - 60 student records
    - Injected missing values  (Age, MathScore, Attendance, Gender)
    - Injected inconsistencies (impossible scores and attendance)
    - Injected outliers        (StudyHrs too high, SciScore too low)
    Saves to CSV file and returns DataFrame.
    """
    np.random.seed(seed)

    data = {
        "StudentID" : range(1, n+1),
        "Age"       : np.random.randint(17, 23, n).astype(float),
        "Gender"    : np.random.choice(["Male", "Female"], n, p=[0.50, 0.50]),
        "Attendance": np.random.uniform(55, 100, n).round(1),
        "MathScore" : np.random.randint(40, 100, n).astype(float),
        "SciScore"  : np.random.randint(40, 100, n).astype(float),
        "StudyHrs"  : np.random.uniform(0.5, 8, n).round(1),
    }

    df = pd.DataFrame(data)

    #Inject Missing Values 
    df.loc[[3, 10, 22],  "Age"]        = np.nan   # 3 missing ages
    df.loc[[5, 15, 30],  "MathScore"]  = np.nan   # 3 missing scores
    df.loc[[7, 20],      "Attendance"] = np.nan   # 2 missing attendance
    df.loc[[1, 33],      "Gender"]     = None     # 2 missing genders

    #Inject Inconsistencies 
    df.loc[0,  "MathScore"]  = 150   # impossible: score > 100
    df.loc[4,  "Attendance"] = -10   # impossible: negative attendance

    #Inject Outliers 
    df.loc[2,  "StudyHrs"]  = 22     # outlier: too high
    df.loc[11, "SciScore"]  = 4      # outlier: extremely low

    #Save to CSV 
    df.to_csv(filename, index=False)
    print(f"Dataset created and saved to '{filename}'")
    print(f"   Rows: {len(df)} | Columns: {list(df.columns)}")

#Generate and Save 
create_academic_dataset("academic_performance.csv", n=60)

#STEP 1 : LOAD DATASET FROM CSV
df = pd.read_csv("academic_performance.csv")

print("\n=== Raw Dataset (first 10 rows) ===")
print(df.head(10))
print("\nShape:", df.shape)

#Q1. Scan for Missing Values & Inconsistencies
print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Inconsistencies Found ===")
print("MathScore > 100 :", df[df["MathScore"] > 100][["StudentID","MathScore"]])
print("Attendance < 0  :", df[df["Attendance"] < 0][["StudentID","Attendance"]])

#Fix inconsistencies → replace impossible values with NaN
df.loc[df["MathScore"]  > 100, "MathScore"]  = np.nan
df.loc[df["Attendance"] < 0,   "Attendance"] = np.nan

#Fill missing values
df["Age"]        = df["Age"].fillna(df["Age"].median())                    # median (numeric)
df["MathScore"]  = df["MathScore"].fillna(df["MathScore"].mean()).round(1) # mean
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].median())      # median
df["Gender"]     = df["Gender"].fillna(df["Gender"].mode()[0])             # mode (categorical)

print("\n=== After Fixing — Missing Values ===")
print(df.isnull().sum())
print("\n=== Clean Data Sample ===")
print(df.head(10))

#Q2. Scan Numeric Variables for Outliers & Handle Them
numeric_cols = ["Age", "Attendance", "MathScore", "SciScore", "StudyHrs"]

print("\n=== Outlier Detection using IQR Method ===")
for col in numeric_cols:
    Q1       = df[col].quantile(0.25)
    Q3       = df[col].quantile(0.75)
    IQR      = Q3 - Q1
    lower    = Q1 - 1.5 * IQR
    upper    = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)][col]
    print(f"{col:12s} | Lower={lower:.2f}  Upper={upper:.2f}  Outliers={list(outliers)}")

#Handle outliers using Capping (Winsorization)
print("\n--- Handling Outliers using Capping (Winsorization) ---")
for col in numeric_cols:
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower, upper=upper)

print("Outliers capped successfully.")
print("StudyHrs max after capping:", df["StudyHrs"].max())
print("SciScore  min after capping:", df["SciScore"].min())

#Q3. Data Transformation (on StudyHrs — right-skewed)
print("\n=== Skewness BEFORE Transformation ===")
print("StudyHrs skewness:", round(df["StudyHrs"].skew(), 4))

#Log Transformation to reduce skewness → approach normal distribution, log1p(x) = log(1+x) — safe even when x = 0
df["StudyHrs_log"] = np.log1p(df["StudyHrs"])

print("StudyHrs skewness AFTER Log Transform:", round(df["StudyHrs_log"].skew(), 4))

print("\n=== Sample: Original vs Log-Transformed StudyHrs ===")
print(df[["StudyHrs", "StudyHrs_log"]].head(10))

print("\n=== Summary Stats: StudyHrs vs StudyHrs_log ===")
print(df[["StudyHrs", "StudyHrs_log"]].describe().round(3))

print("\n=== Final Dataset Shape ===", df.shape)
print("=== Final Columns ===", list(df.columns))
