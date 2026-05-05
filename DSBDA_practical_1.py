# ============================================================
#  DATA SCIENCE PRACTICAL EXAMINATION
#  Dataset : Titanic
#  Source  : https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
# ============================================================


# ── Q1. Import Libraries ────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ── Q2. Dataset Description ─────────────────────────────────
# Name   : Titanic Passenger Dataset
# URL    : https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
# About  : Records of 891 Titanic passengers.
#          Contains survival status, passenger class, name, sex, age,
#          number of siblings/spouses, parents/children aboard, ticket,
#          fare, cabin, and port of embarkation.

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


# ── Q3. Load Dataset ────────────────────────────────────────
df = pd.read_csv(URL)
print("=== First 5 Rows ===")
print(df.head())


# ── Q4. Data Preprocessing ──────────────────────────────────
print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Initial Statistics (describe) ===")
print(df.describe())

print("\n=== Variable Descriptions ===")
# PassengerId - Unique ID          | Survived - 0=No, 1=Yes
# Pclass      - Ticket class 1/2/3 | Name     - Passenger name
# Sex         - male/female        | Age      - Age in years
# SibSp       - Siblings/Spouses   | Parch    - Parents/Children
# Ticket      - Ticket number      | Fare     - Ticket price
# Cabin       - Cabin number       | Embarked - Port (C/Q/S)
print(df.dtypes)

print("\n=== Dimensions (rows, columns) ===")
print(df.shape)


# ── Q5. Data Formatting & Normalization ─────────────────────
# FIX: Direct assignment instead of inplace=True (pandas 2.0 Copy-on-Write fix)
df["Age"]      = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"], inplace=True)

print("\n=== Data Types After Cleaning ===")
print(df.dtypes)
# object  -> Sex, Embarked  (categorical / character)
# float64 -> Age, Fare      (continuous numeric)
# int64   -> Survived, Pclass, SibSp, Parch (integer)

# Min-Max Normalization: scales Age and Fare to 0-1 range
scaler = MinMaxScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])
print("\n=== After Min-Max Normalization (Age & Fare) ===")
print(df[["Age", "Fare"]].head())


# ── Q6. Categorical to Quantitative ─────────────────────────
# Binary Encoding: Sex -> male=0, female=1
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# One-Hot Encoding: Embarked (C/Q/S) -> dummy columns
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

print("\n=== Final DataFrame (Categorical Encoded) ===")
print(df.head())
print("\n=== Final Data Types ===")
print(df.dtypes)
print("\n=== Final Shape ===", df.shape)