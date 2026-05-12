# ============================================================
#  PRACTICAL 8 — Data Visualization : Titanic Dataset
#  1. Explore patterns using Seaborn
#  2. Histogram of Fare Column
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load Titanic Dataset from CSV File ──────────────────────
df = pd.read_csv("Titanic-Dataset.csv")

print("=== Titanic Dataset ===")

print("\nShape :", df.shape)

print("\nFirst 5 Rows :")

print(df.head())

print("\nColumns Present :")

print(df.columns)

# ════════════════════════════════════════════════════════════
# PLOT 1 — HISTOGRAM OF FARE
# ════════════════════════════════════════════════════════════

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Fare"],
    bins=30,
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

# ════════════════════════════════════════════════════════════
# PLOT 2 — COUNT PLOT OF SURVIVAL
# ════════════════════════════════════════════════════════════

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

# ════════════════════════════════════════════════════════════
# PLOT 3 — SCATTER PLOT
# Age vs Fare
# ════════════════════════════════════════════════════════════

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

# ════════════════════════════════════════════════════════════
# Observations
# ════════════════════════════════════════════════════════════

# print("\n=== Observations ===")

# print("""
# 1. Fare distribution is right-skewed.
# 2. Most passengers paid lower fares.
# 3. Female passengers had higher survival rate.
# 4. Higher fare passengers had better survival chances.
# """)
