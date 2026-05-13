#DATA SCIENCE PRACTICAL - Data Wrangling I
#Dataset : Student Performance (Self-Created → data.csv)
#Data Wrangling, I Perform the following operations using Python on any open source dataset (e.g., data.csv) 

#Q1. Import Libraries
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# STEP 0 : CREATE AND SAVE DATASET TO data.csv
def create_student_dataset(filename="data.csv", n=50, seed=42):
    """
    Creates a realistic student performance dataset with:
    - Realistic distributions per grade group
    - Injected missing values
    - Injected inconsistencies
    Saves to CSV file.
    """
    np.random.seed(seed)

    #Grade Groups with realistic score ranges
    grade_configs = {
        "A": {"count": 12, "score": (80, 100), "study": (5.0, 8.0)},
        "B": {"count": 14, "score": (65,  79), "study": (3.5, 6.0)},
        "C": {"count": 13, "score": (50,  64), "study": (2.0, 4.5)},
        "D": {"count": 11, "score": (35,  49), "study": (0.5, 3.0)},
    }

    rows = []
    sid  = 1

    for grade, cfg in grade_configs.items():
        cnt = cfg["count"]
        for _ in range(cnt):
            rows.append({
                "StudentID"  : sid,
                "Age"        : int(np.random.randint(17, 23)),
                "Gender"     : np.random.choice(["Male", "Female"],
                                                p=[0.52, 0.48]),
                "Attendance" : round(np.random.uniform(60, 100), 1),
                "StudyHrs"   : round(np.random.uniform(*cfg["study"]), 1),
                "Score"      : int(np.random.randint(*cfg["score"])),
                "Grade_Group": grade,
            })
            sid += 1

    df = pd.DataFrame(rows)

    #Inject Missing Values 
    df.loc[[2, 9, 18],  "Age"]      = np.nan   # 3 missing ages
    df.loc[[4, 14, 29], "Score"]    = np.nan   # 3 missing scores
    df.loc[[7, 22],     "StudyHrs"] = np.nan   # 2 missing study hours
    df.loc[[1, 33],     "Gender"]   = None     # 2 missing genders

    #Inject Inconsistencies
    df.loc[0,  "Score"]      = 150   # impossible: score > 100
    df.loc[3,  "Attendance"] = -5    # impossible: negative attendance
    df.loc[5,  "Age"]        = 99    # impossible: age too high

    #Save to CSV 
    df.to_csv(filename, index=False)
    print(f"Dataset created and saved to '{filename}'")
    print(f"   Rows: {len(df)} | Columns: {list(df.columns)}")
    return df

create_student_dataset("data.csv", n=50)

#Q2. LOAD DATASET
df = pd.read_csv("data.csv")
print("\n=== First 5 Rows ===")
print(df.head())
print("\n=== Shape ===")
print(df.shape)
print("\n=== Column Names ===")
print(df.columns.tolist())

#Q3. DATA PREPROCESSING
print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Initial Statistics (describe) ===")
print(df.describe().round(2))

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== Variable Descriptions ===")
var_desc = {
    "StudentID"  : "Unique student identifier (integer)",
    "Age"        : "Student age in years (numeric, float)",
    "Gender"     : "Male / Female (categorical)",
    "Attendance" : "Class attendance percentage (float)",
    "StudyHrs"   : "Daily study hours (float)",
    "Score"      : "Exam score out of 100 (numeric)",
    "Grade_Group": "Performance grade A/B/C/D (categorical)",
}
for col, desc in var_desc.items():
    print(f"  {col:12s} : {desc}")

#Q4. FIX INCONSISTENCIES
print("\n=== Inconsistencies Before Fix ===")
print("Score > 100   :", df[df["Score"]    > 100][["StudentID", "Score"]].values)
print("Attendance < 0:", df[df["Attendance"]< 0 ][["StudentID", "Attendance"]].values)
print("Age > 30      :", df[df["Age"]       > 30][["StudentID", "Age"]].values)

#Replace impossible values with NaN
df.loc[df["Score"]      > 100, "Score"]      = np.nan
df.loc[df["Attendance"] < 0,   "Attendance"] = np.nan
df.loc[df["Age"]        > 30,  "Age"]        = np.nan

#Fill Missing Values 
df["Age"]        = df["Age"].fillna(df["Age"].median())
df["Score"]      = df["Score"].fillna(df["Score"].mean()).round(1)
df["StudyHrs"]   = df["StudyHrs"].fillna(df["StudyHrs"].median())
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].median())
df["Gender"]     = df["Gender"].fillna(df["Gender"].mode()[0])

print("\n=== Missing Values After Fix ===")
print(df.isnull().sum())

#Q5. DATA FORMATTING & NORMALIZATION
print("\n=== Data Types After Cleaning ===")
print(df.dtypes)

#Min-Max Normalization on numeric columns
scaler = MinMaxScaler()
df[["Age", "Score", "StudyHrs", "Attendance"]] = scaler.fit_transform(
    df[["Age", "Score", "StudyHrs", "Attendance"]]
)

print("\n=== After Min-Max Normalization ===")
print(df[["Age", "Score", "StudyHrs", "Attendance"]].head())

#Q6. CATEGORICAL TO QUANTITATIVE
#Binary Encoding: Gender → Male=0, Female=1
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

#One-Hot Encoding: Grade_Group → dummy columns
df = pd.get_dummies(df, columns=["Grade_Group"], drop_first=True)

print("\n=== Final DataFrame (Encoded) ===")
print(df.head())
print("\n=== Final Data Types ===")
print(df.dtypes)
print("\n=== Final Shape ===", df.shape)
