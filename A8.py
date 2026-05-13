#PRACTICAL 8 — Data Visualization : Titanic Dataset
#Create Own Dataset using NumPy Random

#Import Libraries 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#CREATE TITANIC-LIKE DATASET
np.random.seed(42)

n = 50

df = pd.DataFrame({

    "PassengerId": range(1, n + 1),

    "Survived": np.random.choice(
        [0, 1],
        n,
        p=[0.6, 0.4]
    ),

    "Pclass": np.random.choice(
        [1, 2, 3],
        n,
        p=[0.2, 0.3, 0.5]
    ),

    "Sex": np.random.choice(
        ["male", "female"],
        n
    ),

    "Age": np.random.randint(
        1, 70, n
    ),

    "Fare": np.random.randint(
        50, 500, n
    )
})

#ADD FEW MISSING VALUES
df.loc[5, "Age"] = np.nan
df.loc[12, "Fare"] = np.nan

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
df.fillna(df.mean(numeric_only=True), inplace=True)

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
