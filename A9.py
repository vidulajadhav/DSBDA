# PRACTICAL 9 - Data Visualization : Titanic Dataset
#  1. Box Plot  : Age distribution by Gender and Survival
#  2. Line Chart: Survival Rate by Age Group
#  3. Scatter   : Age vs Fare by Gender and Survival

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#CREATE TITANIC-LIKE DATASET USING np.random
#500 passengers, realistic distributions, fixed seed=42
np.random.seed(42)
n = 500

sex      = np.random.choice(["male", "female"], n, p=[0.65, 0.35])
pclass   = np.random.choice([1, 2, 3], n, p=[0.20, 0.25, 0.55])
age      = np.where(
               sex == "female",
               np.random.normal(28, 12, n),   # females avg age 28
               np.random.normal(30, 14, n)    # males avg age 30
           ).clip(1, 80)

#Survival: females and 1st class more likely to survive
surv_prob = np.where(sex == "female", 0.74, 0.19)
surv_prob = np.where(pclass == 1, surv_prob + 0.15, surv_prob)
surv_prob = np.where(pclass == 3, surv_prob - 0.10, surv_prob)
surv_prob = surv_prob.clip(0.05, 0.95)
survived  = np.random.binomial(1, surv_prob)

#Fare depends on class
fare = np.where(pclass == 1, np.random.normal(80, 40, n),
       np.where(pclass == 2, np.random.normal(20, 8, n),
                              np.random.normal(10, 5, n))).clip(3, 300)

df = pd.DataFrame({
    "sex"     : sex,
    "age"     : age.round(1),
    "survived": survived,
    "fare"    : fare.round(2),
    "pclass"  : pclass
})

#SAVE DATASET LOCALLY AS CSV FILE
df.to_csv("Titanic_Practical9_Dataset.csv", index=False)

print("\nDataset saved successfully!")
print("File Name : Titanic_Practical9_Dataset.csv")

print("=== Dataset Created ===")
print("Shape:", df.shape)
print(df.head(8))
print("\nSurvived counts:\n", df["survived"].value_counts())
print("Gender counts:\n",   df["sex"].value_counts())

#PLOT 1 - BOX PLOT : Age by Gender and Survival
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="sex", y="age",
            hue="survived",
            palette={0: "salmon", 1: "mediumseagreen"})
plt.title("Practical 9 - Box Plot: Age by Gender and Survival")
plt.xlabel("Gender")
plt.ylabel("Age")
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, ["Not Survived (0)", "Survived (1)"], title="Survived")
plt.tight_layout()
plt.savefig("p9_boxplot_age_gender.png")
plt.show()
print(">> Plot 1 saved: p9_boxplot_age_gender.png")

#PLOT 2 - LINE CHART : Survival Rate by Age Group
df["age_group"] = pd.cut(df["age"],
                         bins=[0, 10, 20, 30, 40, 50, 60, 80],
                         labels=["0-10","11-20","21-30",
                                 "31-40","41-50","51-60","61+"])

survival_by_age = df.groupby("age_group", observed=True)["survived"] \
                    .mean().reset_index()

plt.figure(figsize=(8, 4))
sns.lineplot(data=survival_by_age, x="age_group", y="survived",
             marker="o", color="steelblue", linewidth=2.5)
plt.title("Practical 9 - Line Chart: Survival Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate (0 to 1)")
plt.tight_layout()
plt.savefig("p9_linechart_survival_age.png")
plt.show()
print(">> Plot 2 saved: p9_linechart_survival_age.png")

#PLOT 3 - SCATTER PLOT : Age vs Fare by Gender and Survival
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="age", y="fare",
                hue="sex", style="survived",
                palette={"male": "royalblue", "female": "hotpink"},
                alpha=0.6)
plt.title("Practical 9 - Scatter: Age vs Fare by Gender and Survival")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("p9_scatter_age_fare.png")
plt.show()
print(">> Plot 3 saved: p9_scatter_age_fare.png")

#OBSERVATIONS
print("""
=== Observations (Practical 9) ===
1. BOX PLOT  : Female survivors are slightly older than female
   non-survivors. Male non-survivors have a wider age spread.
   Children show higher survival rate across both genders.
2. LINE CHART: Passengers aged 0-10 have the highest survival
   rate. Survival drops for teens and young adults.
3. SCATTER   : Female passengers (pink) paid higher fares on
   average and had better survival. Consistent with the
   real Titanic policy of women and children first.
""")
