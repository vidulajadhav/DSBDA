# ============================================================
#  PRACTICAL 9 — Data Visualization : Titanic Dataset
#  1. Box Plot: Age distribution by Gender & Survival
#  + Line Chart and Scatter Plot
# ============================================================

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ── Load Dataset ─────────────────────────────────────────────
df = sns.load_dataset("titanic")
df_clean = df.dropna(subset=["age"])   # drop rows where age is missing

# ════════════════════════════════════════════════════════════
# PLOT 1 — BOX PLOT : Age by Gender & Survival
# ════════════════════════════════════════════════════════════
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, x="sex", y="age",
            hue="survived", palette={0:"salmon", 1:"mediumseagreen"})
plt.title("Practical 9 — Box Plot: Age by Gender & Survival")
plt.xlabel("Gender")
plt.ylabel("Age")
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, ["No (0)", "Yes (1)"], title="Survived")
plt.tight_layout()
plt.savefig("p9_boxplot_age_gender.png")
plt.show()
print(">> Plot 1 saved: p9_boxplot_age_gender.png")

# ════════════════════════════════════════════════════════════
# PLOT 2 — LINE CHART : Survival Rate by Age Group
# ════════════════════════════════════════════════════════════
df_clean = df_clean.copy()
df_clean["age_group"] = pd.cut(df_clean["age"],
                                bins=[0,10,20,30,40,50,60,80],
                                labels=["0-10","11-20","21-30",
                                        "31-40","41-50","51-60","61+"])
survival_by_age = df_clean.groupby("age_group", observed=True)["survived"].mean().reset_index()

plt.figure(figsize=(8, 4))
sns.lineplot(data=survival_by_age, x="age_group", y="survived",
             marker="o", color="steelblue", linewidth=2.5)
plt.title("Practical 9 — Line Chart: Survival Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate (0 to 1)")
plt.tight_layout()
plt.savefig("p9_linechart_survival_age.png")
plt.show()
print(">> Plot 2 saved: p9_linechart_survival_age.png")

# ════════════════════════════════════════════════════════════
# PLOT 3 — SCATTER PLOT : Age vs Fare by Gender
# ════════════════════════════════════════════════════════════
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x="age", y="fare",
                hue="sex", style="survived",
                palette={"male":"royalblue","female":"hotpink"},
                alpha=0.6)
plt.title("Practical 9 — Scatter Plot: Age vs Fare by Gender & Survival")
plt.xlabel("Age")
plt.ylabel("Fare (£)")
plt.tight_layout()
plt.savefig("p9_scatter_age_fare_gender.png")
plt.show()
print(">> Plot 3 saved: p9_scatter_age_fare_gender.png")

# ── Observations ────────────────────────────────────────────
print("""
=== Observations (Practical 9) ===
1. BOX PLOT  : Female survivors were slightly older than female
   non-survivors. Male non-survivors had a wider age range.
   Children (low age) show higher survival across both genders.
2. LINE CHART: Children aged 0-10 had the highest survival rate.
   Survival drops for teens and young adults, then varies for
   older age groups.
3. SCATTER   : Female passengers paid slightly higher fares on
   average and had better survival outcomes (especially at
   higher fare ranges), consistent with "women and children first".
""")
