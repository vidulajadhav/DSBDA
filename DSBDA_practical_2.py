# ============================================================
#  DATA SCIENCE PRACTICAL - Academic Performance Dataset
#  Dataset : Created manually (realistic student data)
# ============================================================
# ── Q1. Import Libraries ────────────────────────────────────
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ── Create Dataset ───────────────────────────────────────────
np.random.seed(42)
n = 50

data = {
    "StudentID" : range(1, n+1),
    "Age"       : np.random.randint(17, 23, n).astype(float),
    "Gender"    : np.random.choice(["Male", "Female", None], n, p=[0.48, 0.48, 0.04]),
    "Attendance": np.random.uniform(55, 100, n).round(1),
    "MathScore" : np.random.randint(40, 100, n).astype(float),
    "SciScore"  : np.random.randint(40, 100, n).astype(float),
    "StudyHrs"  : np.random.uniform(0.5, 8, n).round(1),
}

df = pd.DataFrame(data)

# Inject missing values
df.loc[[3, 10, 22], "Age"]        = np.nan
df.loc[[5, 15, 30], "MathScore"]  = np.nan
df.loc[[7, 20],     "Attendance"] = np.nan

# Inject inconsistency (impossible score)
df.loc[0, "MathScore"]  = 150    # score > 100
df.loc[1, "Attendance"] = -10    # negative attendance

# Inject outliers
df.loc[2,  "StudyHrs"]  = 22     # outlier: too high
df.loc[11, "SciScore"]  = 4      # outlier: extremely low

print("=== Raw Dataset (first 10 rows) ===")
print(df.head(10))
print("\nShape:", df.shape)


# ══════════════════════════════════════════════════════════════
# Q1. Scan for Missing Values & Inconsistencies
# ══════════════════════════════════════════════════════════════
print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Inconsistencies Found ===")
print("MathScore > 100 :", df[df["MathScore"] > 100][["StudentID","MathScore"]])
print("Attendance < 0  :", df[df["Attendance"] < 0][["StudentID","Attendance"]])

# Fix inconsistencies → replace impossible values with NaN
df.loc[df["MathScore"]  > 100, "MathScore"]  = np.nan
df.loc[df["Attendance"] < 0,   "Attendance"] = np.nan

# Fill missing values
df["Age"]        = df["Age"].fillna(df["Age"].median())          # median (numeric)
df["MathScore"]  = df["MathScore"].fillna(df["MathScore"].mean()).round(1)  # mean
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].median())
df["Gender"]     = df["Gender"].fillna(df["Gender"].mode()[0])   # mode (categorical)

print("\n=== After Fixing — Missing Values ===")
print(df.isnull().sum())
print("\n=== Clean Data Sample ===")
print(df.head(10))


# ══════════════════════════════════════════════════════════════
# Q2. Scan Numeric Variables for Outliers & Handle Them
# ══════════════════════════════════════════════════════════════
numeric_cols = ["Age", "Attendance", "MathScore", "SciScore", "StudyHrs"]

print("\n=== Outlier Detection using IQR Method ===")
for col in numeric_cols:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)][col]
    print(f"{col:12s} | Lower={lower:.2f}  Upper={upper:.2f}  Outliers={list(outliers)}")

# Handle outliers using Capping (Winsorization)
print("\n--- Handling Outliers using Capping (Winsorization) ---")
for col in numeric_cols:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower, upper=upper)

print("Outliers capped. StudyHrs max after capping:", df["StudyHrs"].max())
print("SciScore min after capping:", df["SciScore"].min())


# ══════════════════════════════════════════════════════════════
# Q3. Data Transformation (on StudyHrs — right-skewed)
# ══════════════════════════════════════════════════════════════
print("\n=== Skewness BEFORE Transformation ===")
print("StudyHrs skewness:", round(df["StudyHrs"].skew(), 4))

# Log Transformation to reduce skewness and approach normal distribution
df["StudyHrs_log"] = np.log1p(df["StudyHrs"])   # log1p = log(1+x), safe for 0 values

print("StudyHrs skewness AFTER Log Transform:", round(df["StudyHrs_log"].skew(), 4))

print("\n=== Sample: Original vs Log-Transformed StudyHrs ===")
print(df[["StudyHrs", "StudyHrs_log"]].head(10))

print("\n=== Summary Stats: StudyHrs vs StudyHrs_log ===")
print(df[["StudyHrs", "StudyHrs_log"]].describe().round(3))

print("\n=== Final Dataset Shape ===", df.shape)
print("=== Final Columns ===", list(df.columns))
