#PRACTICAL 8 — Data Visualization : Titanic Dataset
#Create Own Dataset using NumPy Random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#CREATE TITANIC-LIKE DATASET
np.random.seed(42)
n = 500

sex      = np.random.choice(["male", "female"], n, p=[0.65, 0.35])
pclass   = np.random.choice([1, 2, 3], n, p=[0.20, 0.25, 0.55])
embarked = np.random.choice(["C", "Q", "S"], n, p=[0.20, 0.10, 0.70])

age = np.where(
    sex == "female",
    np.random.normal(28, 12, n),
    np.random.normal(30, 14, n)
).clip(1, 80).round(1)

surv_prob = np.where(sex == "female", 0.74, 0.19)
surv_prob = np.where(pclass == 1, surv_prob + 0.15, surv_prob)
surv_prob = np.where(pclass == 3, surv_prob - 0.10, surv_prob)
surv_prob = surv_prob.clip(0.05, 0.95)
survived  = np.random.binomial(1, surv_prob)

fare = np.where(
    pclass == 1, np.random.normal(300, 80, n),
    np.where(
        pclass == 2, np.random.normal(150, 40, n),
                     np.random.normal(70,  20, n)
    )
).clip(10, 500).round(2)

df = pd.DataFrame({
    "PassengerId": range(1, n + 1),
    "Survived"   : survived,
    "Pclass"     : pclass,
    "Sex"        : sex,
    "Age"        : age,
    "Fare"       : fare,
    "Embarked"   : embarked,
})

#ADD MISSING VALUES
missing_age      = np.random.choice(n, 30, replace=False)
missing_fare     = np.random.choice(n, 10, replace=False)
missing_embarked = np.random.choice(n, 5,  replace=False)

df.loc[missing_age,      "Age"]      = np.nan
df.loc[missing_fare,     "Fare"]     = np.nan
df.loc[missing_embarked, "Embarked"] = np.nan

#SAVE DATASET LOCALLY
df.to_csv("Titanic-Dataset.csv", index=False)
print("Dataset created and saved successfully!")

#LOAD DATASET
df = pd.read_csv("Titanic-Dataset.csv")

#DISPLAY DATASET
print("\n=== Titanic Dataset ===")
print("\nShape :", df.shape)
print("\nFirst 5 Rows :")
print(df.head())
print("\nColumns Present :")
print(df.columns)

#CHECK MISSING VALUES
print("\n=== Missing Values Before Handling ===")
print(df.isnull().sum())

#HANDLE MISSING VALUES
df["Age"]      = df["Age"].fillna(df["Age"].median())
df["Fare"]     = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("\n=== Missing Values After Handling ===")
print(df.isnull().sum())

#PLOT 1 — HISTOGRAM OF FARE
plt.figure(figsize=(8, 5))
sns.histplot(
    df["Fare"],
    bins=10,
    kde=True,
    color="steelblue"
)
plt.title("Histogram : Distribution of Fare")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("histogram_fare.png")
plt.show()
print("\nPlot 1 Saved : histogram_fare.png")

#PLOT 2 — COUNT PLOT OF SURVIVAL
plt.figure(figsize=(6, 4))
sns.countplot(
    data=df,
    x="Survived",
    hue="Sex"
)
plt.title("Survival Count by Gender")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Passenger Count")
plt.tight_layout()
plt.savefig("survival_countplot.png")
plt.show()
print("Plot 2 Saved : survival_countplot.png")

#PLOT 3 — SCATTER PLOT Age vs Fare
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Age",
    y="Fare",
    hue="Survived",
    alpha=0.7
)
plt.title("Scatter Plot : Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("scatter_age_fare.png")
plt.show()
print("Plot 3 Saved : scatter_age_fare.png")

#OBSERVATIONS
print("\n=== Observations ===")
print("""
1. Fare distribution is slightly right-skewed.
2. Most passengers paid medium-range fares.
3. Female passengers show better survival counts.
4. Higher fare passengers appear to have better survival chances.
""")

#CREATE MANUAL TITANIC DATASET
# data = {

#     "PassengerId": list(range(1, 31)),

#     "Survived": [
#         0,1,1,0,1,0,0,1,1,0,
#         1,0,0,1,1,0,1,0,0,1,
#         1,0,1,0,1,0,1,0,0,1
#     ],

#     "Pclass": [
#         3,1,2,3,1,3,2,1,2,3,
#         1,3,2,1,2,3,1,3,2,1,
#         2,3,1,3,1,2,1,3,2,1
#     ],

#     "Sex": [
#         "male","female","female","male","female",
#         "male","male","female","female","male",
#         "female","male","male","female","female",
#         "male","female","male","male","female",
#         "female","male","female","male","female",
#         "male","female","male","male","female"
#     ],

#     "Age": [
#         22,38,26,35,28,
#         30,40,19,24,32,
#         27,45,50,21,29,
#         34,18,41,36,23,
#         25,48,20,39,31,
#         44,17,33,42,26
#     ],

#     "Fare": [
#         120,350,220,90,400,
#         80,150,500,270,100,
#         380,70,140,450,260,
#         95,520,110,160,300,
#         240,85,470,130,360,
#         170,550,105,145,320
#     ]
# }
